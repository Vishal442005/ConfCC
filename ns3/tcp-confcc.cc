/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * ConfCC – Conformal Congestion Control  (implementation)
 */
#include "tcp-confcc.h"
#include "ns3/log.h"
#include "ns3/simulator.h"
#include "ns3/tcp-socket-state.h"
#include "ns3/double.h"
#include "ns3/uinteger.h"
#include "ns3/boolean.h"
#include "ns3/string.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace ns3 {

NS_LOG_COMPONENT_DEFINE ("TcpConfCC");
NS_OBJECT_ENSURE_REGISTERED (TcpConfCC);

// ======================================================================
// AdaptiveConformalInterval
// ======================================================================

AdaptiveConformalInterval::AdaptiveConformalInterval (double alpha,
                                                      double gamma,
                                                      uint32_t window)
  : m_alpha (alpha),
    m_gamma (gamma),
    m_alphaT (alpha),
    m_window (window),
    m_nSamples (0),
    m_nCovered (0),
    m_quantile (0.0),
    m_sumScores (0.0)
{
}

bool AdaptiveConformalInterval::Update (double predicted_ms, double observed_ms)
{
  double score = observed_ms - predicted_ms;  // non-conformity score
  bool covered = (score <= m_quantile);

  // ACI update: alpha_{t+1} = alpha_t + gamma * (alpha - 1{not covered})
  double err = covered ? 0.0 : 1.0;
  m_alphaT = std::clamp (m_alphaT + m_gamma * (m_alpha - err), 0.001, 0.999);

  // Maintain sliding calibration window
  m_scores.push_back (score);
  m_sumScores += score;
  if (m_scores.size () > m_window)
    {
      m_sumScores -= m_scores.front ();
      m_scores.pop_front ();
    }

  m_nSamples++;
  if (covered) m_nCovered++;

  RecomputeQuantile ();
  return covered;
}

double AdaptiveConformalInterval::UpperBound (double predicted_ms) const
{
  return predicted_ms + m_quantile;
}

double AdaptiveConformalInterval::GetCoverage () const
{
  if (m_nSamples == 0) return 1.0;
  return static_cast<double> (m_nCovered) / m_nSamples;
}

void AdaptiveConformalInterval::RecomputeQuantile ()
{
  if (m_scores.empty ())
    {
      m_quantile = 0.0;
      return;
    }
  // (1-alpha_t) empirical quantile of calibration window
  std::vector<double> sorted (m_scores.begin (), m_scores.end ());
  std::sort (sorted.begin (), sorted.end ());
  double q_level = 1.0 - m_alphaT;
  size_t idx = static_cast<size_t> (std::ceil (q_level * sorted.size ())) - 1;
  idx = std::min (idx, sorted.size () - 1);
  m_quantile = std::max (0.0, sorted[idx]);
}

void AdaptiveConformalInterval::Reset ()
{
  m_alphaT = m_alpha;
  m_nSamples = 0;
  m_nCovered = 0;
  m_scores.clear ();
  m_quantile = 0.0;
  m_sumScores = 0.0;
}

// ======================================================================
// TcpConfCC
// ======================================================================

TypeId TcpConfCC::GetTypeId ()
{
  static TypeId tid =
    TypeId ("ns3::TcpConfCC")
      .SetParent<TcpCongestionOps> ()
      .SetGroupName ("Internet")
      .AddConstructor<TcpConfCC> ()
      // ACI parameters
      .AddAttribute ("Alpha", "Target miscoverage rate (0 < alpha < 1).",
                     DoubleValue (0.05),
                     MakeDoubleAccessor (&TcpConfCC::m_alpha),
                     MakeDoubleChecker<double> (0.001, 0.5))
      .AddAttribute ("Gamma", "ACI adaptation step size.",
                     DoubleValue (0.005),
                     MakeDoubleAccessor (&TcpConfCC::m_gamma),
                     MakeDoubleChecker<double> (1e-6, 0.5))
      .AddAttribute ("CalibrationWindow",
                     "Number of samples in the sliding calibration window.",
                     UintegerValue (500),
                     MakeUintegerAccessor (&TcpConfCC::m_calWindow),
                     MakeUintegerChecker<uint32_t> (10))
      // Control law parameters
      .AddAttribute ("TargetRttMs",
                     "Desired standing RTT in milliseconds.",
                     DoubleValue (50.0),
                     MakeDoubleAccessor (&TcpConfCC::m_targetRttMs),
                     MakeDoubleChecker<double> (1.0, 10000.0))
      .AddAttribute ("EwmaGain",
                     "Gain for EWMA RTT predictor (0 < g <= 1).",
                     DoubleValue (0.125),
                     MakeDoubleAccessor (&TcpConfCC::m_ewmaGain),
                     MakeDoubleChecker<double> (0.001, 1.0))
      .AddAttribute ("RttHistLen",
                     "Length of RTT history for predictor seeding.",
                     UintegerValue (8),
                     MakeUintegerAccessor (&TcpConfCC::m_rttHistLen),
                     MakeUintegerChecker<uint32_t> (1))
      // Logging
      .AddAttribute ("OutputFile",
                     "CSV file path for per-ACK trace (empty = no file).",
                     StringValue (""),
                     MakeStringAccessor (&TcpConfCC::m_outFile),
                     MakeStringChecker ())
      // TracedValues
      .AddTraceSource ("Cwnd", "Congestion window (segments)",
                       MakeTraceSourceAccessor (&TcpConfCC::m_trCwnd),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("RttMs", "Last observed RTT (ms)",
                       MakeTraceSourceAccessor (&TcpConfCC::m_trRttMs),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("UpperBound", "ACI upper bound on RTT (ms)",
                       MakeTraceSourceAccessor (&TcpConfCC::m_trUpperBound),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("AlphaT", "Current adaptive alpha_t",
                       MakeTraceSourceAccessor (&TcpConfCC::m_trAlphaT),
                       "ns3::TracedValueCallback::Double")
      .AddTraceSource ("Covered", "1 if last RTT <= upper bound",
                       MakeTraceSourceAccessor (&TcpConfCC::m_trCovered),
                       "ns3::TracedValueCallback::Int");
  return tid;
}

TcpConfCC::TcpConfCC ()
  : m_alpha (0.05),
    m_gamma (0.005),
    m_calWindow (500),
    m_targetRttMs (50.0),
    m_ewmaGain (0.125),
    m_rttHistLen (8),
    m_ssThreshOnLoss (true),
    m_aci (0.05, 0.005, 500),
    m_ewmaRtt (0.0),
    m_ewmaInit (false),
    m_lastRttMs (0.0),
    m_upperBound (0.0),
    m_inSlowStart (true),
    m_bytesAckedEpoch (0),
    m_lastEpochTime (Seconds (0)),
    m_targetDelayOffsetMs (10.0),
    m_minRttMs (1e9),
    m_trCwnd (0.0),
    m_trRttMs (0.0),
    m_trUpperBound (0.0),
    m_trAlphaT (0.05),
    m_trCovered (0),
    m_trGoodput (0.0),
    m_headerWritten (false)
{
  NS_LOG_FUNCTION (this);
}

TcpConfCC::TcpConfCC (const TcpConfCC &sock)
  : TcpCongestionOps (sock),
    m_alpha (sock.m_alpha),
    m_gamma (sock.m_gamma),
    m_calWindow (sock.m_calWindow),
    m_targetRttMs (sock.m_targetRttMs),
    m_ewmaGain (sock.m_ewmaGain),
    m_rttHistLen (sock.m_rttHistLen),
    m_ssThreshOnLoss (sock.m_ssThreshOnLoss),
    m_aci (sock.m_alpha, sock.m_gamma, sock.m_calWindow),
    m_rttHistory (sock.m_rttHistory),
    m_ewmaRtt (sock.m_ewmaRtt),
    m_ewmaInit (sock.m_ewmaInit),
    m_lastRttMs (sock.m_lastRttMs),
    m_upperBound (sock.m_upperBound),
    m_inSlowStart (sock.m_inSlowStart),
    m_bytesAckedEpoch (sock.m_bytesAckedEpoch),
    m_lastEpochTime (sock.m_lastEpochTime),
    m_targetDelayOffsetMs (sock.m_targetDelayOffsetMs),
    m_minRttMs (sock.m_minRttMs),
    m_trCwnd (0.0),
    m_trRttMs (0.0),
    m_trUpperBound (0.0),
    m_trAlphaT (sock.m_alpha),
    m_trCovered (0),
    m_trGoodput (0.0),
    m_outFile (sock.m_outFile),
    m_headerWritten (false)
{
}

TcpConfCC::~TcpConfCC ()
{
}

void TcpConfCC::Init (Ptr<TcpSocketState> tcb)
{
  NS_LOG_FUNCTION (this << tcb);
  m_aci = AdaptiveConformalInterval (m_alpha, m_gamma, m_calWindow);
  m_ewmaInit = false;
  m_ewmaRtt = 0.0;
  m_rttHistory.clear ();
  m_inSlowStart = true;
  m_bytesAckedEpoch = 0;
  m_lastEpochTime = Simulator::Now ();
  m_minRttMs = 1e9;

  // Write CSV header if output path set
  if (!m_outFile.empty () && !m_headerWritten)
    {
      std::ofstream f (m_outFile, std::ios::out | std::ios::trunc);
      f << "t_ms,cwnd,rtt_ms,qdelay_ms,upper_bound_ms,alpha_t,covered,goodput_bps,flow_id\n";
      m_headerWritten = true;
    }
}

void TcpConfCC::PktsAcked (Ptr<TcpSocketState> tcb,
                            uint32_t            segmentsAcked,
                            const Time         &rtt)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked << rtt);

  if (rtt.IsZero ()) return;

  double rtt_ms = rtt.GetSeconds () * 1000.0;
  m_lastRttMs   = rtt_ms;
  if (rtt_ms < m_minRttMs)
    m_minRttMs = rtt_ms;

  // Predict next RTT using EWMA
  double predicted_ms = PredictNextRtt ();

  // Update ACI with the observed RTT
  bool covered = m_aci.Update (predicted_ms, rtt_ms);

  // Compute ACI upper bound for the *next* window adjustment
  m_upperBound = m_aci.UpperBound (predicted_ms);

  // Update EWMA *after* ACI update (EWMA uses current sample)
  UpdateRttHistory (rtt_ms);

  // Update traced values
  m_trRttMs    = rtt_ms;
  m_trUpperBound = m_upperBound;
  m_trAlphaT   = m_aci.GetAlphaT ();
  m_trCovered  = covered ? 1 : 0;
  m_trCwnd     = static_cast<double> (tcb->m_cWnd.Get ()) / tcb->m_segmentSize;

  // Write CSV row
  if (!m_outFile.empty ())
    {
      WriteRow (0); // flow_id placeholder; scenario sets it via attribute
    }
}

void TcpConfCC::IncreaseWindow (Ptr<TcpSocketState> tcb,
                                uint32_t            segmentsAcked)
{
  NS_LOG_FUNCTION (this << tcb << segmentsAcked);

  uint32_t seg = tcb->m_segmentSize;
  if (seg == 0) seg = 1448;

  if (m_inSlowStart)
    {
      // Standard slow-start (same as NewReno) until we have calibration data
      if (m_aci.GetSampleCount () < 20)
        {
          tcb->m_cWnd += seg * segmentsAcked;
          return;
        }
      m_inSlowStart = false;
    }

  if (m_upperBound <= 0.0 || m_targetRttMs <= 0.0)
    {
      // Standard additive increase fallback
      double adder = static_cast<double> (seg * seg) / tcb->m_cWnd.Get ();
      adder = std::max (1.0, adder);
      tcb->m_cWnd += static_cast<uint32_t> (adder * segmentsAcked);
      return;
    }

  // Accumulate acked bytes for RTT epoch pacing
  m_bytesAckedEpoch += segmentsAcked * seg;

  // Perform window adjustment once per RTT (or when acked bytes exceed cwnd)
  Time now = Simulator::Now ();
  Time minInterval = (m_lastRttMs > 0.0) ? MilliSeconds (m_lastRttMs * 0.8) : MilliSeconds (20);
  if (m_bytesAckedEpoch >= tcb->m_cWnd.Get () || (now - m_lastEpochTime) >= minInterval)
    {
      m_bytesAckedEpoch = 0;
      m_lastEpochTime = now;

      // Compute dynamic delay target: base propagation delay + bounded queue margin
      double effTarget = m_targetRttMs;
      if (m_minRttMs > 0.0 && m_minRttMs < 1e8)
        {
          // Allow up to m_targetDelayOffsetMs (10ms) of queueing above minRTT,
          // or user-specified targetRttMs, whichever is larger to avoid path underutilization
          effTarget = std::max (m_targetRttMs, m_minRttMs + m_targetDelayOffsetMs);
        }

      double ratio = effTarget / m_upperBound;

      if (ratio >= 1.0)
        {
          // Conformal upper bound is under delay target: safe to probe bandwidth.
          // Additive increase scaled by headroom: (ratio - 1.0) bonus up to 2 segments.
          double bonus = std::min (2.0, (ratio - 1.0) * 2.0);
          tcb->m_cWnd += static_cast<uint32_t> ((1.0 + bonus) * seg);
        }
      else
        {
          // Conformal upper bound exceeds target: gentle multiplicative backoff per RTT
          // Clamp decrement so window never drops by more than 15% in a single RTT
          double backoff = std::clamp (ratio, 0.85, 0.98);
          uint32_t new_cwnd = static_cast<uint32_t> (tcb->m_cWnd.Get () * backoff);
          tcb->m_cWnd = std::max (2 * seg, new_cwnd);
        }
    }
}

uint32_t TcpConfCC::GetSsThresh (Ptr<const TcpSocketState> tcb,
                                  uint32_t                  bytesInFlight)
{
  // Halve cwnd on loss (same as NewReno), but do not go below 2 segments
  uint32_t seg = tcb->m_segmentSize;
  return std::max (2 * seg, tcb->m_cWnd.Get () / 2);
}

void TcpConfCC::CongestionStateSet (Ptr<TcpSocketState> tcb,
                                    const TcpSocketState::TcpCongState_t newState)
{
  if (newState == TcpSocketState::CA_LOSS)
    {
      m_inSlowStart = true;
      // Reset ACI on severe congestion so it relearns the new regime
      m_aci.Reset ();
    }
}

Ptr<TcpCongestionOps> TcpConfCC::Fork ()
{
  return CopyObject<TcpConfCC> (this);
}

// -----------------------------------------------------------------------
// Private helpers
// -----------------------------------------------------------------------

double TcpConfCC::PredictNextRtt ()
{
  if (!m_ewmaInit)
    {
      // Seed from history mean if available, else return 0
      if (!m_rttHistory.empty ())
        {
          m_ewmaRtt  = std::accumulate (m_rttHistory.begin (),
                                        m_rttHistory.end (), 0.0)
                       / m_rttHistory.size ();
          m_ewmaInit = true;
        }
      return m_ewmaRtt;
    }
  return m_ewmaRtt;
}

void TcpConfCC::UpdateRttHistory (double rtt_ms)
{
  if (!m_ewmaInit)
    {
      m_ewmaRtt  = rtt_ms;
      m_ewmaInit = true;
    }
  else
    {
      m_ewmaRtt = (1.0 - m_ewmaGain) * m_ewmaRtt + m_ewmaGain * rtt_ms;
    }

  m_rttHistory.push_back (rtt_ms);
  while (m_rttHistory.size () > m_rttHistLen)
    m_rttHistory.pop_front ();
}

void TcpConfCC::WriteRow (uint32_t flowId)
{
  std::ofstream f (m_outFile, std::ios::app);
  double t_ms = Simulator::Now ().GetSeconds () * 1000.0;
  double cwnd_segs = static_cast<double> (m_trCwnd);
  // qdelay = RTT - base RTT (approx as RTT - target)
  double qdelay = std::max (0.0, m_lastRttMs - m_targetRttMs);
  f << std::fixed << std::setprecision (3)
    << t_ms << ","
    << cwnd_segs << ","
    << m_lastRttMs << ","
    << qdelay << ","
    << m_upperBound << ","
    << m_aci.GetAlphaT () << ","
    << static_cast<int> (m_trCovered) << ","
    << 0.0 /* goodput placeholder */ << ","
    << flowId << "\n";
}

} // namespace ns3


// ======================================================================
// TcpDelayTargetNoACI  ──  Ablation: same control law, no ACI
// ======================================================================
namespace ns3 {

NS_OBJECT_ENSURE_REGISTERED (TcpDelayTargetNoACI);

TypeId TcpDelayTargetNoACI::GetTypeId ()
{
  static TypeId tid = TypeId ("ns3::TcpDelayTargetNoACI")
    .SetParent<TcpCongestionOps> ()
    .SetGroupName ("Internet")
    .AddConstructor<TcpDelayTargetNoACI> ()
    .AddAttribute ("TargetRttMs",
                   "Target standing RTT (ms) for delay-target control",
                   DoubleValue (50.0),
                   MakeDoubleAccessor (&TcpDelayTargetNoACI::m_targetRttMs),
                   MakeDoubleChecker<double> (1.0, 2000.0))
    .AddAttribute ("FixedMarginMs",
                   "Fixed upper-bound margin added to EWMA RTT (replaces ACI quantile)",
                   DoubleValue (10.0),
                   MakeDoubleAccessor (&TcpDelayTargetNoACI::m_fixedMarginMs),
                   MakeDoubleChecker<double> (0.0, 500.0))
    .AddAttribute ("EwmaGain",
                   "EWMA filter gain for RTT prediction",
                   DoubleValue (0.1),
                   MakeDoubleAccessor (&TcpDelayTargetNoACI::m_ewmaGain),
                   MakeDoubleChecker<double> (0.001, 1.0));
  return tid;
}

TcpDelayTargetNoACI::TcpDelayTargetNoACI ()
  : TcpCongestionOps (),
    m_targetRttMs   (50.0),
    m_fixedMarginMs (10.0),
    m_ewmaGain      (0.1),
    m_ewmaRtt       (0.0),
    m_ewmaInit      (false),
    m_inSlowStart   (true),
    m_bytesAckedEpoch (0),
    m_lastEpochTime (Seconds (0)),
    m_targetDelayOffsetMs (10.0),
    m_minRttMs (1e9)
{}

TcpDelayTargetNoACI::TcpDelayTargetNoACI (const TcpDelayTargetNoACI &sock)
  : TcpCongestionOps (sock),
    m_targetRttMs   (sock.m_targetRttMs),
    m_fixedMarginMs (sock.m_fixedMarginMs),
    m_ewmaGain      (sock.m_ewmaGain),
    m_ewmaRtt       (sock.m_ewmaRtt),
    m_ewmaInit      (sock.m_ewmaInit),
    m_inSlowStart   (sock.m_inSlowStart),
    m_bytesAckedEpoch (sock.m_bytesAckedEpoch),
    m_lastEpochTime (sock.m_lastEpochTime),
    m_targetDelayOffsetMs (sock.m_targetDelayOffsetMs),
    m_minRttMs (sock.m_minRttMs)
{}

void TcpDelayTargetNoACI::Init (Ptr<TcpSocketState> tcb)
{
  m_ewmaInit    = false;
  m_inSlowStart = true;
  m_bytesAckedEpoch = 0;
  m_lastEpochTime = Simulator::Now ();
  m_minRttMs = 1e9;
}

void TcpDelayTargetNoACI::PktsAcked (Ptr<TcpSocketState> tcb,
                                      uint32_t            segmentsAcked,
                                      const Time         &rtt)
{
  if (rtt.IsZero ()) return;
  double rtt_ms = rtt.GetSeconds () * 1000.0;
  if (rtt_ms < m_minRttMs)
    m_minRttMs = rtt_ms;

  if (!m_ewmaInit)
    {
      m_ewmaRtt  = rtt_ms;
      m_ewmaInit = true;
    }
  else
    {
      m_ewmaRtt = (1.0 - m_ewmaGain) * m_ewmaRtt + m_ewmaGain * rtt_ms;
    }

  // Slow-start exit: leave SS once we observe RTT inflation
  if (m_inSlowStart && rtt_ms > m_targetRttMs * 0.9)
    m_inSlowStart = false;
}

void TcpDelayTargetNoACI::IncreaseWindow (Ptr<TcpSocketState> tcb,
                                          uint32_t             segmentsAcked)
{
  uint32_t seg = tcb->m_segmentSize;
  if (seg == 0) seg = 1448;

  if (m_inSlowStart)
    {
      // Slow-start: standard additive increase
      tcb->m_cWnd += seg * segmentsAcked;
      return;
    }

  m_bytesAckedEpoch += segmentsAcked * seg;
  Time now = Simulator::Now ();
  Time minInterval = (m_ewmaRtt > 0.0) ? MilliSeconds (m_ewmaRtt * 0.8) : MilliSeconds (20);

  if (m_bytesAckedEpoch >= tcb->m_cWnd.Get () || (now - m_lastEpochTime) >= minInterval)
    {
      m_bytesAckedEpoch = 0;
      m_lastEpochTime = now;

      double effTarget = m_targetRttMs;
      if (m_minRttMs > 0.0 && m_minRttMs < 1e8)
        {
          effTarget = std::max (m_targetRttMs, m_minRttMs + m_targetDelayOffsetMs);
        }

      // Fixed upper bound = EWMA + fixed margin (no ACI)
      double upper_bound = m_ewmaRtt + m_fixedMarginMs;
      double ratio       = effTarget / upper_bound;

      if (ratio >= 1.0)
        {
          double bonus = std::min (2.0, (ratio - 1.0) * 2.0);
          tcb->m_cWnd += static_cast<uint32_t> ((1.0 + bonus) * seg);
        }
      else
        {
          double backoff = std::clamp (ratio, 0.85, 0.98);
          uint32_t new_cwnd = static_cast<uint32_t> (tcb->m_cWnd.Get () * backoff);
          tcb->m_cWnd = std::max (2 * seg, new_cwnd);
        }
    }
}

uint32_t TcpDelayTargetNoACI::GetSsThresh (Ptr<const TcpSocketState> tcb,
                                            uint32_t                   bytesInFlight)
{
  uint32_t seg = tcb->m_segmentSize;
  return std::max (2 * seg, tcb->m_cWnd.Get () / 2);
}

void TcpDelayTargetNoACI::CongestionStateSet (
    Ptr<TcpSocketState>              tcb,
    const TcpSocketState::TcpCongState_t newState)
{
  if (newState == TcpSocketState::CA_LOSS)
    m_inSlowStart = true;
}

Ptr<TcpCongestionOps> TcpDelayTargetNoACI::Fork ()
{
  return CopyObject<TcpDelayTargetNoACI> (this);
}

} // namespace ns3
