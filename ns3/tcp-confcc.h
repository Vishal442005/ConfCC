/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * ConfCC – Conformal Congestion Control
 *
 * Derives from TcpCongestionOps.  The key idea is to wrap any delay-target
 * controller with an Adaptive Conformal Interval (ACI) predictor so that
 * the one-step-ahead RTT upper bound carries a statistical coverage guarantee
 * at level 1-α even under distribution shift.
 *
 * Paper: "ConfCC: Statistically-Safe Congestion Control via Adaptive
 *         Conformal Prediction" (in preparation, 2025-26)
 */
#ifndef TCP_CONFCC_H
#define TCP_CONFCC_H

#include "ns3/tcp-congestion-ops.h"
#include "ns3/traced-value.h"
#include "ns3/data-rate.h"
#include <deque>
#include <vector>

namespace ns3 {

/**
 * \brief Online Adaptive Conformal Interval tracker.
 *
 * Maintains a sliding calibration window of non-conformity scores
 * (residuals between predicted and observed RTT) and updates the
 * inflation factor alpha_t via the ACI update rule so that the
 * long-run empirical coverage converges to 1-alpha.
 */
class AdaptiveConformalInterval
{
public:
  /**
   * \param alpha   target miscoverage rate (e.g. 0.05 for 95% coverage)
   * \param gamma   ACI step-size (controls adaptation speed, e.g. 0.005)
   * \param window  calibration window length in samples
   */
  AdaptiveConformalInterval (double alpha = 0.05,
                             double gamma = 0.005,
                             uint32_t window = 500);

  /** Feed a new (prediction, observation) pair; returns coverage flag. */
  bool Update (double predicted_ms, double observed_ms);

  /** Current upper bound = predicted_ms + quantile_inflation */
  double UpperBound (double predicted_ms) const;

  /** Current alpha_t (miscoverage level being targeted) */
  double GetAlphaT () const { return m_alphaT; }

  /** Running empirical coverage ratio */
  double GetCoverage () const;

  /** Number of calibration samples collected so far */
  uint32_t GetSampleCount () const { return m_nSamples; }

  void Reset ();

private:
  double   m_alpha;          // target miscoverage (fixed)
  double   m_gamma;          // ACI learning rate
  double   m_alphaT;         // current adaptive alpha
  uint32_t m_window;         // sliding window size
  uint32_t m_nSamples;       // total samples seen
  uint32_t m_nCovered;       // total covered samples

  std::deque<double> m_scores;   // sliding window of non-conformity scores
  double             m_quantile; // current (1-alphaT) empirical quantile
  double             m_sumScores;// sum for O(1) quantile update approximation

  void RecomputeQuantile ();
};

// ---------------------------------------------------------------------------

/**
 * \brief ConfCC congestion control algorithm.
 *
 * Control law:
 *   cwnd_{t+1} = cwnd_t * min(1, target_rtt / upper_bound_t)
 *
 * where upper_bound_t is the ACI-covered RTT prediction.  During slow-start
 * the algorithm behaves identically to TcpNewReno.
 */
class TcpConfCC : public TcpCongestionOps
{
public:
  static TypeId GetTypeId ();
  TcpConfCC ();
  TcpConfCC (const TcpConfCC &sock);
  ~TcpConfCC () override;

  std::string GetName () const override { return "TcpConfCC"; }

  // ----- TcpCongestionOps interface -----
  void Init (Ptr<TcpSocketState> tcb) override;
  void PktsAcked (Ptr<TcpSocketState> tcb,
                  uint32_t segmentsAcked,
                  const Time &rtt) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void CongestionStateSet (Ptr<TcpSocketState> tcb,
                           const TcpSocketState::TcpCongState_t newState) override;
  Ptr<TcpCongestionOps> Fork () override;

private:
  // ----- Predictor helpers -----
  double PredictNextRtt (); // EWMA of recent RTT samples
  void   UpdateRttHistory (double rtt_ms);

  // ----- Parameters (exposed as ns-3 attributes) -----
  double   m_alpha;          // ACI miscoverage target
  double   m_gamma;          // ACI learning rate
  uint32_t m_calWindow;      // ACI calibration window
  double   m_targetRttMs;    // desired standing RTT (ms)
  double   m_ewmaGain;       // gain for EWMA RTT predictor
  uint32_t m_rttHistLen;     // RTT history length for predictor
  bool     m_ssThreshOnLoss; // use TcpNewReno ssthresh on loss?

  // ----- State -----
  AdaptiveConformalInterval m_aci;
  std::deque<double> m_rttHistory;
  double   m_ewmaRtt;        // EWMA predicted RTT (ms)
  bool     m_ewmaInit;       // first sample flag
  double   m_lastRttMs;      // last observed RTT
  double   m_upperBound;     // last ACI upper bound
  bool     m_inSlowStart;

  // ----- TracedValues (one CSV row per ACK) -----
  TracedValue<double> m_trCwnd;
  TracedValue<double> m_trRttMs;
  TracedValue<double> m_trUpperBound;
  TracedValue<double> m_trAlphaT;
  TracedValue<int>    m_trCovered;
  TracedValue<double> m_trGoodput;

  // ----- Pacing & RTT epoch tracking -----
  uint32_t m_bytesAckedEpoch;
  Time     m_lastEpochTime;
  double   m_targetDelayOffsetMs; // offset above minRTT (ms)
  double   m_minRttMs;            // tracked minimum RTT (ms)

  // ----- CSV logging -----
  std::string m_outFile;
  bool        m_headerWritten;
  void        WriteRow (uint32_t flowId);
};

/**
 * \brief Ablation: Delay-Target controller WITHOUT Adaptive Conformal Inference.
 *
 * Identical control law to TcpConfCC (multiplicative window scaling toward
 * target RTT) but uses a fixed EWMA predictor with no conformal calibration.
 * The upper bound is simply:  U_t = EWMA_t + fixed_margin_ms
 *
 * This is the direct ablation baseline that isolates the contribution of ACI.
 * Including this scheme lets reviewers see whether conformal calibration adds
 * value beyond a plain heuristic delay-target controller.
 */
class TcpDelayTargetNoACI : public TcpCongestionOps
{
public:
  static TypeId GetTypeId ();
  TcpDelayTargetNoACI ();
  TcpDelayTargetNoACI (const TcpDelayTargetNoACI &sock);
  ~TcpDelayTargetNoACI () override = default;

  std::string GetName () const override { return "TcpDelayTargetNoACI"; }

  void Init (Ptr<TcpSocketState> tcb) override;
  void PktsAcked (Ptr<TcpSocketState> tcb,
                  uint32_t segmentsAcked,
                  const Time &rtt) override;
  void IncreaseWindow (Ptr<TcpSocketState> tcb,
                       uint32_t segmentsAcked) override;
  uint32_t GetSsThresh (Ptr<const TcpSocketState> tcb,
                        uint32_t bytesInFlight) override;
  void CongestionStateSet (Ptr<TcpSocketState> tcb,
                           const TcpSocketState::TcpCongState_t newState) override;
  Ptr<TcpCongestionOps> Fork () override;

private:
  double   m_targetRttMs;   // desired standing RTT (ms)
  double   m_fixedMarginMs; // fixed upper-bound margin (replaces ACI quantile)
  double   m_ewmaGain;      // EWMA gain
  double   m_ewmaRtt;       // current EWMA RTT
  bool     m_ewmaInit;
  bool     m_inSlowStart;
  uint32_t m_bytesAckedEpoch;
  Time     m_lastEpochTime;
  double   m_targetDelayOffsetMs;
  double   m_minRttMs;
};

} // namespace ns3
#endif /* TCP_CONFCC_H */
