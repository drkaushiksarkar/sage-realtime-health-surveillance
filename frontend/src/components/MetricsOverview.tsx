import React from "react";

interface Props {
  metrics: {
    eventsPerSecond: number;
    avgLatencyMs: number;
    p99LatencyMs: number;
    activeAlerts: number;
    countriesMonitored: number;
  };
}

export function MetricsOverview({ metrics }: Props) {
  const cards = [
    { label: "Events/sec", value: metrics.eventsPerSecond.toLocaleString(), color: "text-green-400" },
    { label: "Avg Latency", value: `${metrics.avgLatencyMs.toFixed(1)}ms`, color: metrics.avgLatencyMs < 150 ? "text-green-400" : "text-red-400" },
    { label: "P99 Latency", value: `${metrics.p99LatencyMs.toFixed(1)}ms`, color: metrics.p99LatencyMs < 200 ? "text-green-400" : "text-red-400" },
    { label: "Active Alerts", value: metrics.activeAlerts.toString(), color: "text-yellow-400" },
    { label: "Countries", value: metrics.countriesMonitored.toString(), color: "text-blue-400" },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {cards.map((c) => (
        <div key={c.label} className="rounded-lg border border-gray-800 bg-gray-900 p-3">
          <div className="text-xs text-gray-400">{c.label}</div>
          <div className={`text-2xl font-bold ${c.color}`}>{c.value}</div>
        </div>
      ))}
    </div>
  );
}
