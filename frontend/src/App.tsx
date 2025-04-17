import React, { useEffect, useState } from "react";
import { AlertPanel } from "./components/AlertPanel";
import { MetricsOverview } from "./components/MetricsOverview";
import { AnomalyMap } from "./components/AnomalyMap";
import { LatencyMonitor } from "./components/LatencyMonitor";

interface SystemMetrics {
  eventsPerSecond: number;
  avgLatencyMs: number;
  p99LatencyMs: number;
  activeAlerts: number;
  countriesMonitored: number;
}

export default function App() {
  const [metrics, setMetrics] = useState<SystemMetrics>({
    eventsPerSecond: 0, avgLatencyMs: 0, p99LatencyMs: 0,
    activeAlerts: 0, countriesMonitored: 194,
  });

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/metrics");
    ws.onmessage = (event) => setMetrics(JSON.parse(event.data));
    return () => ws.close();
  }, []);

  return (
    <div className="min-h-screen bg-gray-950 text-white">
      <header className="border-b border-gray-800 px-6 py-4">
        <h1 className="text-xl font-bold">SAGE Real-Time Health Surveillance</h1>
        <p className="text-sm text-gray-400">
          Processing {metrics.eventsPerSecond.toLocaleString()} events/sec |
          Latency: {metrics.avgLatencyMs.toFixed(1)}ms avg, {metrics.p99LatencyMs.toFixed(1)}ms p99
        </p>
      </header>
      <main className="grid grid-cols-12 gap-4 p-6">
        <div className="col-span-8">
          <AnomalyMap />
        </div>
        <div className="col-span-4 space-y-4">
          <MetricsOverview metrics={metrics} />
          <LatencyMonitor budgetMs={200} currentMs={metrics.avgLatencyMs} />
        </div>
        <div className="col-span-12">
          <AlertPanel />
        </div>
      </main>
    </div>
  );
}
