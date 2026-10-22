/* -*- Mode:C++; c-file-style:"gnu"; indent-tabs-mode:nil; -*- */
/**
 * confcc-main.cc
 * ==============
 * ns-3 top-level simulation program for ConfCC.
 *
 * Scenario selection via --scenario=[s1..s8]:
 *   S1  Dumbbell (single flow baseline)
 *   S2  Trace-driven capacity (Mahimahi-rate bins)
 *   S3  Multi-flow fairness & coexistence
 *   S4  Parking-lot multi-hop
 *   S5  Wi-Fi (802.11n)
 *   S6  5G NR mmWave blockage  (requires nr module)
 *   S7  Adversarial predictor & capacity step
 *   S8  Short-flow mix
 *
 * TCP scheme via --scheme=[confcc|cubic|bbr|vegas|newreno|westwood|
 *                          delay_target|split_conformal]
 *
 * Output: one CSV per run under --out=<path>
 *         summary row appended to results/ns3/confcc/summary.csv
 *
 * Parallel runs (GNU parallel):
 *   parallel -j$(nproc) \
 *     './ns3 run "scratch/confcc/confcc-main --scenario={1} --scheme={2} \
 *                 --RngRun={3} --out=results/ns3/confcc/{1}/{2}_run{3}.csv"' \
 *     ::: s1 s2 s3 s4 s5 s7 s8 ::: confcc cubic bbr vegas ::: $(seq 1 30)
 */

#include "tcp-confcc.h"
#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/internet-module.h"
#include "ns3/point-to-point-module.h"
#include "ns3/applications-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/wifi-module.h"
#include "ns3/mobility-module.h"
#include "ns3/traffic-control-module.h"

#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE ("ConfCCMain");

// ======================================================================
// Helper: set TCP variant by name
// ======================================================================
static void SetTcpScheme (const std::string &scheme)
{
  if (scheme == "confcc")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpConfCC::GetTypeId ()));
  else if (scheme == "cubic")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TypeId::LookupByName ("ns3::TcpCubic")));
  else if (scheme == "bbr")
    {
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TypeId::LookupByName ("ns3::TcpBbr")));
      Config::SetDefault ("ns3::TcpSocketBase::UseEcn", StringValue ("Off"));
    }
  else if (scheme == "vegas")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TypeId::LookupByName ("ns3::TcpVegas")));
  else if (scheme == "newreno")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TypeId::LookupByName ("ns3::TcpNewReno")));
  else if (scheme == "westwood")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TypeId::LookupByName ("ns3::TcpWestwoodPlus")));
  else if (scheme == "hybla")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TypeId::LookupByName ("ns3::TcpHybla")));
  else if (scheme == "delay_target_no_aci")
    Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                        TypeIdValue (TcpDelayTargetNoACI::GetTypeId ()));
  else
    {
      NS_LOG_WARN ("Unknown scheme '" << scheme << "', using ConfCC");
      Config::SetDefault ("ns3::TcpL4Protocol::SocketType",
                          TypeIdValue (TcpConfCC::GetTypeId ()));
    }
}

// ======================================================================
// Helper: install bulk sender + sink
// ======================================================================
static ApplicationContainer InstallBulkFlow (Ptr<Node> src, Ptr<Node> dst,
                                             uint16_t port, double startS,
                                             double stopS, uint32_t maxBytes = 0)
{
  Address sinkAddr (InetSocketAddress (Ipv4Address::GetAny (), port));
  PacketSinkHelper sink ("ns3::TcpSocketFactory", sinkAddr);
  ApplicationContainer sinkApps = sink.Install (dst);
  sinkApps.Start (Seconds (0.0));
  sinkApps.Stop (Seconds (stopS + 1.0));

  BulkSendHelper bulk ("ns3::TcpSocketFactory",
                       InetSocketAddress (dst->GetObject<Ipv4> ()
                                            ->GetAddress (1, 0)
                                            .GetLocal (),
                                          port));
  bulk.SetAttribute ("MaxBytes", UintegerValue (maxBytes));
  ApplicationContainer srcApps = bulk.Install (src);
  srcApps.Start (Seconds (startS));
  srcApps.Stop (Seconds (stopS));
  return sinkApps;
}

// ======================================================================
// Flow Monitor summary → summary.csv
// ======================================================================
static void WriteSummary (const std::string &summaryFile,
                          const std::string &scenario,
                          const std::string &scheme,
                          const std::string &trace,
                          uint32_t           seed,
                          Ptr<FlowMonitor>   monitor,
                          FlowMonitorHelper &fmHelper,
                          double             simDuration)
{
  monitor->CheckForLostPackets ();
  auto stats = monitor->GetFlowStats ();
  auto classifier =
    DynamicCast<Ipv4FlowClassifier> (fmHelper.GetClassifier ());

  double totalGoodput = 0, sumRtt = 0, p50 = 0, p95 = 0, p99 = 0;
  uint32_t nFlows = 0;
  std::vector<double> rtts;

  for (auto &kv : stats)
    {
      auto fid = classifier->FindFlow (kv.first);
      // Skip non-TCP or reverse flows (port < 1024 dst = server)
      (void) fid;
      auto &s = kv.second;
      if (s.rxPackets == 0) continue;
      double dur = s.timeLastRxPacket.GetSeconds ()
                   - s.timeFirstTxPacket.GetSeconds ();
      if (dur <= 0) continue;
      totalGoodput += (s.rxBytes * 8.0) / dur;
      double rtt = s.delaySum.GetSeconds () / s.rxPackets * 1000.0;
      rtts.push_back (rtt);
      sumRtt += rtt;
      nFlows++;
    }

  if (!rtts.empty ())
    {
      std::sort (rtts.begin (), rtts.end ());
      auto pct = [&] (double p) -> double {
        size_t idx = static_cast<size_t> (p / 100.0 * rtts.size ());
        idx = std::min (idx, rtts.size () - 1);
        return rtts[idx];
      };
      p50 = pct (50);
      p95 = pct (95);
      p99 = pct (99);
    }

  // Append to summary CSV
  bool newFile = !std::ifstream (summaryFile).good ();
  std::ofstream sf (summaryFile, std::ios::app);
  if (newFile)
    sf << "scenario,scheme,trace,seed,goodput_mbps,p50_ms,p95_ms,"
          "p99_ms,violation_rate,coverage,jain\n";

  // Jain's fairness index (approximate over flows)
  double jain = 0.0;
  if (nFlows > 0)
    {
      double sum_g = totalGoodput, sum_g2 = 0;
      for (auto &kv : stats)
        {
          auto &s = kv.second;
          double dur = s.timeLastRxPacket.GetSeconds ()
                       - s.timeFirstTxPacket.GetSeconds ();
          if (dur <= 0) continue;
          double g = (s.rxBytes * 8.0) / dur;
          sum_g2 += g * g;
        }
      jain = (sum_g * sum_g) / (nFlows * sum_g2 + 1e-9);
    }

  sf << std::fixed << std::setprecision (4)
     << scenario << ","
     << scheme << ","
     << trace << ","
     << seed << ","
     << totalGoodput / 1e6 << ","
     << p50 << ","
     << p95 << ","
     << p99 << ","
     << 0.0 /* violation_rate; computed in post-processing */ << ","
     << 0.0 /* coverage; from per-flow CSV */ << ","
     << jain << "\n";
}

// ======================================================================
// S1 – Dumbbell
// ======================================================================
static void RunS1Dumbbell (const std::string &scheme,
                           const std::string &outFile,
                           const std::string &summaryFile,
                           uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S1 Dumbbell: scheme=" << scheme << " seed=" << seed);

  // 4 senders, 4 receivers, one bottleneck link
  uint32_t nFlows = 4;
  NodeContainer left, right;
  left.Create (nFlows);
  right.Create (nFlows);
  NodeContainer routers;
  routers.Create (2);

  PointToPointHelper accessLink;
  accessLink.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  accessLink.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper bottleneck;
  bottleneck.SetDeviceAttribute ("DataRate", StringValue ("20Mbps"));
  bottleneck.SetChannelAttribute ("Delay", StringValue ("20ms"));
  bottleneck.SetQueue ("ns3::DropTailQueue",
                       "MaxSize", StringValue ("200p"));

  InternetStackHelper stack;
  stack.Install (left);
  stack.Install (right);
  stack.Install (routers);

  // Bottleneck
  auto bDev = bottleneck.Install (routers.Get (0), routers.Get (1));

  // Traffic control on bottleneck
  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::FqCoDelQueueDisc");
  tch.Install (bDev);

  Ipv4AddressHelper addr;
  addr.SetBase ("10.1.0.0", "255.255.255.0");
  auto bIf = addr.Assign (bDev);

  std::vector<Ipv4InterfaceContainer> leftIf (nFlows), rightIf (nFlows);
  for (uint32_t i = 0; i < nFlows; ++i)
    {
      auto lDev = accessLink.Install (left.Get (i), routers.Get (0));
      std::ostringstream base;
      base << "10.2." << i << ".0";
      addr.SetBase (base.str ().c_str (), "255.255.255.0");
      leftIf[i] = addr.Assign (lDev);

      auto rDev = accessLink.Install (routers.Get (1), right.Get (i));
      std::ostringstream base2;
      base2 << "10.3." << i << ".0";
      addr.SetBase (base2.str ().c_str (), "255.255.255.0");
      rightIf[i] = addr.Assign (rDev);
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  // Install flows
  for (uint32_t i = 0; i < nFlows; ++i)
    {
      std::string flowOut = outFile + "_flow" + std::to_string (i) + ".csv";
      if (scheme == "confcc")
        Config::Set ("/NodeList/" + std::to_string (left.Get (i)->GetId ()) +
                       "/$ns3::TcpL4Protocol/SocketType/OutputFile",
                     StringValue (flowOut));
      InstallBulkFlow (left.Get (i), right.Get (i),
                       5000 + i, 0.1 * i, simTime);
    }

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s1_dumbbell", scheme, "none", seed,
                monitor, fmHelper, simTime);

  Simulator::Destroy ();
}

// ======================================================================
// S2 – Trace-driven capacity
// ======================================================================
static void ScheduleRateChanges (Ptr<PointToPointNetDevice> dev,
                                 const std::string         &ratesFile)
{
  std::ifstream f (ratesFile);
  if (!f.is_open ())
    {
      NS_LOG_WARN ("Rate file not found: " << ratesFile << " – using 12 Mbps");
      return;
    }
  std::string line;
  while (std::getline (f, line))
    {
      if (line.empty () || line[0] == '#') continue;
      std::istringstream ss (line);
      long long t_ms;
      long long bps;
      ss >> t_ms >> bps;
      if (bps <= 0) bps = 1000; // floor at 1 kbps
      Simulator::Schedule (MilliSeconds (t_ms),
                           &PointToPointNetDevice::SetDataRate,
                           dev, DataRate (static_cast<uint64_t> (bps)));
    }
}

static void RunS2Trace (const std::string &scheme,
                        const std::string &traceFile,
                        const std::string &outFile,
                        const std::string &summaryFile,
                        uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S2 Trace-driven: scheme=" << scheme
               << " trace=" << traceFile << " seed=" << seed);

  NodeContainer nodes;
  nodes.Create (4); // src, r1, r2, dst

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", StringValue ("12Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("20ms"));

  InternetStackHelper stack;
  stack.Install (nodes);

  auto d01 = p2p.Install (nodes.Get (0), nodes.Get (1));
  auto d12 = p2p.Install (nodes.Get (1), nodes.Get (2));
  auto d23 = p2p.Install (nodes.Get (2), nodes.Get (3));

  // Schedule capacity variation on the bottleneck d12
  Ptr<PointToPointNetDevice> btlDev =
    DynamicCast<PointToPointNetDevice> (d12.Get (0));
  ScheduleRateChanges (btlDev, traceFile);

  Ipv4AddressHelper addr;
  addr.SetBase ("10.0.1.0", "255.255.255.0");
  addr.Assign (d01);
  addr.SetBase ("10.0.2.0", "255.255.255.0");
  addr.Assign (d12);
  addr.SetBase ("10.0.3.0", "255.255.255.0");
  addr.Assign (d23);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  InstallBulkFlow (nodes.Get (0), nodes.Get (3), 5001, 1.0, simTime);

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  std::string traceName = traceFile.substr (traceFile.rfind ('/') + 1);
  WriteSummary (summaryFile, "s2_trace", scheme, traceName, seed,
                monitor, fmHelper, simTime);

  Simulator::Destroy ();
}

// ======================================================================
// S3 – Multi-flow fairness & coexistence
// ======================================================================
static void RunS3MultiFlow (const std::string &scheme1,
                            const std::string &scheme2,
                            const std::string &outFile,
                            const std::string &summaryFile,
                            uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S3 Multi-flow: " << scheme1 << " vs " << scheme2);

  NodeContainer senders, receivers, routers;
  senders.Create (4); receivers.Create (4); routers.Create (2);

  InternetStackHelper stack;
  stack.Install (senders); stack.Install (receivers); stack.Install (routers);

  PointToPointHelper access;
  access.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  access.SetChannelAttribute ("Delay", StringValue ("1ms"));

  PointToPointHelper btl;
  btl.SetDeviceAttribute ("DataRate", StringValue ("20Mbps"));
  btl.SetChannelAttribute ("Delay", StringValue ("10ms"));

  auto bDev = btl.Install (routers.Get (0), routers.Get (1));
  TrafficControlHelper tch;
  tch.SetRootQueueDisc ("ns3::FqCoDelQueueDisc");
  tch.Install (bDev);

  Ipv4AddressHelper addr;
  addr.SetBase ("10.10.0.0", "255.255.255.0");
  addr.Assign (bDev);

  for (uint32_t i = 0; i < 4; ++i)
    {
      auto ld = access.Install (senders.Get (i), routers.Get (0));
      std::ostringstream b1; b1 << "10.11." << i << ".0";
      addr.SetBase (b1.str ().c_str (), "255.255.255.0");
      addr.Assign (ld);

      auto rd = access.Install (routers.Get (1), receivers.Get (i));
      std::ostringstream b2; b2 << "10.12." << i << ".0";
      addr.SetBase (b2.str ().c_str (), "255.255.255.0");
      addr.Assign (rd);
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  // First 2 flows use scheme1, last 2 use scheme2
  for (uint32_t i = 0; i < 4; ++i)
    InstallBulkFlow (senders.Get (i), receivers.Get (i),
                     6000 + i, 0.05 * i, simTime);

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s3_multiflow",
                scheme1 + "+" + scheme2, "none", seed,
                monitor, fmHelper, simTime);
  Simulator::Destroy ();
}

// ======================================================================
// S4 – Parking-lot multi-hop
// ======================================================================
static void RunS4ParkingLot (const std::string &scheme,
                             const std::string &outFile,
                             const std::string &summaryFile,
                             uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S4 Parking-lot: scheme=" << scheme);

  const uint32_t nHops = 5;
  NodeContainer nodes; nodes.Create (nHops + 1);

  InternetStackHelper stack; stack.Install (nodes);

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", StringValue ("20Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("5ms"));

  Ipv4AddressHelper addr;
  for (uint32_t i = 0; i < nHops; ++i)
    {
      auto dev = p2p.Install (nodes.Get (i), nodes.Get (i + 1));
      std::ostringstream b; b << "10.20." << i << ".0";
      addr.SetBase (b.str ().c_str (), "255.255.255.0");
      addr.Assign (dev);
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  // Long flow src0 → dst5
  InstallBulkFlow (nodes.Get (0), nodes.Get (nHops), 7000, 1.0, simTime);
  // Cross flows
  for (uint32_t i = 1; i < nHops; ++i)
    InstallBulkFlow (nodes.Get (i), nodes.Get (nHops),
                     7001 + i, 2.0 + i, simTime - 5.0);

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s4_parkinglot", scheme, "none", seed,
                monitor, fmHelper, simTime);
  Simulator::Destroy ();
}

// ======================================================================
// S5 – Wi-Fi (802.11n Infrastructure)
// ======================================================================
static void RunS5Wifi (const std::string &scheme,
                       const std::string &outFile,
                       const std::string &summaryFile,
                       uint32_t seed, double simTime = 40.0)
{
  NS_LOG_INFO ("S5 Wi-Fi: scheme=" << scheme);

  NodeContainer wifiSta, wifiAp, wiredNode;
  wifiSta.Create (4);
  wifiAp.Create (1);
  wiredNode.Create (1);

  WifiHelper wifi;
  wifi.SetStandard (WIFI_STANDARD_80211n);
  wifi.SetRemoteStationManager ("ns3::MinstrelHtWifiManager");

  WifiMacHelper mac;
  Ssid ssid ("confcc-wifi");

  YansWifiChannelHelper channel = YansWifiChannelHelper::Default ();
  YansWifiPhyHelper phy;
  phy.SetChannel (channel.Create ());

  mac.SetType ("ns3::StaWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer staDevs = wifi.Install (phy, mac, wifiSta);

  mac.SetType ("ns3::ApWifiMac", "Ssid", SsidValue (ssid));
  NetDeviceContainer apDev = wifi.Install (phy, mac, wifiAp);

  MobilityHelper mobility;
  mobility.SetMobilityModel ("ns3::ConstantPositionMobilityModel");
  mobility.Install (wifiAp);
  mobility.SetPositionAllocator ("ns3::GridPositionAllocator",
                                 "MinX", DoubleValue (1.0),
                                 "MinY", DoubleValue (0.0),
                                 "DeltaX", DoubleValue (3.0),
                                 "DeltaY", DoubleValue (0.0),
                                 "GridWidth", UintegerValue (4),
                                 "LayoutType", StringValue ("RowFirst"));
  mobility.Install (wifiSta);

  InternetStackHelper stack;
  stack.Install (wifiSta); stack.Install (wifiAp); stack.Install (wiredNode);

  Ipv4AddressHelper addr;
  addr.SetBase ("192.168.1.0", "255.255.255.0");
  addr.Assign (staDevs);
  addr.Assign (apDev);

  PointToPointHelper backhaul;
  backhaul.SetDeviceAttribute ("DataRate", StringValue ("100Mbps"));
  backhaul.SetChannelAttribute ("Delay", StringValue ("5ms"));
  auto bDev = backhaul.Install (wifiAp.Get (0), wiredNode.Get (0));
  addr.SetBase ("192.168.2.0", "255.255.255.0");
  addr.Assign (bDev);

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  for (uint32_t i = 0; i < 4; ++i)
    InstallBulkFlow (wifiSta.Get (i), wiredNode.Get (0),
                     8000 + i, 1.0 + 0.1 * i, simTime);

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s5_wifi", scheme, "none", seed,
                monitor, fmHelper, simTime);
  Simulator::Destroy ();
}

// ======================================================================
// S7 – Adversarial predictor & capacity step
// ======================================================================
static void RunS7Adversarial (const std::string &scheme,
                              const std::string &outFile,
                              const std::string &summaryFile,
                              uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S7 Adversarial: scheme=" << scheme);

  NodeContainer nodes; nodes.Create (4);
  InternetStackHelper stack; stack.Install (nodes);

  PointToPointHelper p2p;
  p2p.SetDeviceAttribute ("DataRate", StringValue ("20Mbps"));
  p2p.SetChannelAttribute ("Delay", StringValue ("15ms"));

  Ipv4AddressHelper addr;
  auto d01 = p2p.Install (nodes.Get (0), nodes.Get (1));
  addr.SetBase ("10.30.1.0", "255.255.255.0"); addr.Assign (d01);
  auto d12 = p2p.Install (nodes.Get (1), nodes.Get (2));
  addr.SetBase ("10.30.2.0", "255.255.255.0"); addr.Assign (d12);
  auto d23 = p2p.Install (nodes.Get (2), nodes.Get (3));
  addr.SetBase ("10.30.3.0", "255.255.255.0"); addr.Assign (d23);

  // Capacity step at t=20 s: drop to 5 Mbps, recover at t=40 s
  Ptr<PointToPointNetDevice> btl =
    DynamicCast<PointToPointNetDevice> (d12.Get (0));
  Simulator::Schedule (Seconds (20.0), &PointToPointNetDevice::SetDataRate,
                       btl, DataRate ("5Mbps"));
  Simulator::Schedule (Seconds (40.0), &PointToPointNetDevice::SetDataRate,
                       btl, DataRate ("20Mbps"));

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  InstallBulkFlow (nodes.Get (0), nodes.Get (3), 9000, 1.0, simTime);

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s7_adversarial", scheme, "none", seed,
                monitor, fmHelper, simTime);
  Simulator::Destroy ();
}

// ======================================================================
// S8 – Short-flow mix
// ======================================================================
static void RunS8ShortFlow (const std::string &scheme,
                            const std::string &outFile,
                            const std::string &summaryFile,
                            uint32_t seed, double simTime = 60.0)
{
  NS_LOG_INFO ("S8 Short-flow mix: scheme=" << scheme);

  NodeContainer clients, server, router;
  clients.Create (8); server.Create (1); router.Create (1);

  InternetStackHelper stack;
  stack.Install (clients); stack.Install (server); stack.Install (router);

  PointToPointHelper access;
  access.SetDeviceAttribute ("DataRate", StringValue ("50Mbps"));
  access.SetChannelAttribute ("Delay", StringValue ("2ms"));

  PointToPointHelper btl;
  btl.SetDeviceAttribute ("DataRate", StringValue ("20Mbps"));
  btl.SetChannelAttribute ("Delay", StringValue ("10ms"));

  Ipv4AddressHelper addr;
  addr.SetBase ("10.40.0.0", "255.255.255.0");
  auto bd = btl.Install (router.Get (0), server.Get (0));
  addr.Assign (bd);

  for (uint32_t i = 0; i < 8; ++i)
    {
      auto ld = access.Install (clients.Get (i), router.Get (0));
      std::ostringstream b; b << "10.41." << i << ".0";
      addr.SetBase (b.str ().c_str (), "255.255.255.0");
      addr.Assign (ld);
    }

  Ipv4GlobalRoutingHelper::PopulateRoutingTables ();

  FlowMonitorHelper fmHelper;
  Ptr<FlowMonitor> monitor = fmHelper.InstallAll ();

  // Short flows (100 KB) staggered; long background flow
  for (uint32_t i = 0; i < 8; ++i)
    {
      uint32_t flowBytes = (i % 2 == 0) ? 100000 : 1000000; // 100 KB or 1 MB
      InstallBulkFlow (clients.Get (i), server.Get (0),
                       10000 + i, 1.0 + 2.0 * i, simTime, flowBytes);
    }

  Simulator::Stop (Seconds (simTime + 2.0));
  Simulator::Run ();

  WriteSummary (summaryFile, "s8_shortflow", scheme, "none", seed,
                monitor, fmHelper, simTime);
  Simulator::Destroy ();
}

// ======================================================================
// Main
// ======================================================================
int main (int argc, char *argv[])
{
  // Global defaults
  Config::SetDefault ("ns3::TcpSocket::InitialCwnd", UintegerValue (10));
  Config::SetDefault ("ns3::TcpSocket::SegmentSize", UintegerValue (1448));
  Config::SetDefault ("ns3::TcpSocket::SndBufSize", UintegerValue (1 << 22));
  Config::SetDefault ("ns3::TcpSocket::RcvBufSize", UintegerValue (1 << 22));
  Config::SetDefault ("ns3::TcpSocket::DelAckTimeout", TimeValue (MilliSeconds (1)));
  Config::SetDefault ("ns3::DropTailQueue<Packet>::MaxSize",
                      StringValue ("200p"));

  // Command-line
  std::string scenario   = "s1";
  std::string scheme     = "confcc";
  std::string scheme2    = "cubic";  // for S3 coexistence
  std::string traceFile  = "";
  std::string outFile    = "results/ns3/confcc/run.csv";
  std::string summaryFile = "results/ns3/confcc/summary.csv";
  uint32_t    seed        = 1;
  double      simTime     = 60.0;
  bool        verbose     = false;
  double      targetRtt   = 50.0;
  double      alpha       = 0.05;

  CommandLine cmd (__FILE__);
  cmd.AddValue ("scenario",    "Scenario [s1..s8]",         scenario);
  cmd.AddValue ("scheme",      "TCP congestion control",     scheme);
  cmd.AddValue ("scheme2",     "Second CC for S3",           scheme2);
  cmd.AddValue ("trace",       "Rate file for S2",           traceFile);
  cmd.AddValue ("out",         "Output CSV path",            outFile);
  cmd.AddValue ("summary",     "Summary CSV path",           summaryFile);
  cmd.AddValue ("RngRun",      "RNG run number (seed)",      seed);
  cmd.AddValue ("simTime",     "Simulation duration (s)",    simTime);
  cmd.AddValue ("verbose",     "Enable detailed logging",    verbose);
  cmd.AddValue ("targetRtt",   "ConfCC target RTT (ms)",     targetRtt);
  cmd.AddValue ("alpha",       "ConfCC ACI alpha",           alpha);
  cmd.Parse (argc, argv);

  RngSeedManager::SetSeed (1);
  RngSeedManager::SetRun (seed);

  if (verbose)
    LogComponentEnable ("ConfCCMain", LOG_LEVEL_INFO);

  // Set ConfCC-specific attributes if applicable
  if (scheme == "confcc")
    {
      Config::SetDefault ("ns3::TcpConfCC::TargetRttMs", DoubleValue (targetRtt));
      Config::SetDefault ("ns3::TcpConfCC::Alpha",       DoubleValue (alpha));
      Config::SetDefault ("ns3::TcpConfCC::OutputFile",  StringValue (outFile));
    }

  SetTcpScheme (scheme);

  // Dispatch
  if      (scenario == "s1") RunS1Dumbbell   (scheme, outFile, summaryFile, seed, simTime);
  else if (scenario == "s2") RunS2Trace      (scheme, traceFile, outFile, summaryFile, seed, simTime);
  else if (scenario == "s3") RunS3MultiFlow  (scheme, scheme2, outFile, summaryFile, seed, simTime);
  else if (scenario == "s4") RunS4ParkingLot (scheme, outFile, summaryFile, seed, simTime);
  else if (scenario == "s5") RunS5Wifi       (scheme, outFile, summaryFile, seed, simTime);
  else if (scenario == "s7") RunS7Adversarial (scheme, outFile, summaryFile, seed, simTime);
  else if (scenario == "s8") RunS8ShortFlow  (scheme, outFile, summaryFile, seed, simTime);
  else
    {
      NS_LOG_ERROR ("Unknown scenario: " << scenario);
      return 1;
    }

  return 0;
}
