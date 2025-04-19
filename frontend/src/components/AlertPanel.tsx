import React, { useEffect, useState } from "react";

interface Alert {
  alert_id: string;
  severity: "critical" | "high" | "warning" | "info";
  country_code: string;
  indicator_code: string;
  anomaly_score: number;
  value: number;
  timestamp: string;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: "bg-red-900 border-red-500",
  high: "bg-orange-900 border-orange-500",
  warning: "bg-yellow-900 border-yellow-500",
  info: "bg-blue-900 border-blue-500",
};

export function AlertPanel() {
  const [alerts, setAlerts] = useState<Alert[]>([]);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws/alerts");
    ws.onmessage = (event) => {
      const alert: Alert = JSON.parse(event.data);
      setAlerts((prev) => [alert, ...prev].slice(0, 50));
    };
    return () => ws.close();
  }, []);

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900 p-4">
      <h2 className="mb-3 text-lg font-semibold">Active Alerts</h2>
      <div className="space-y-2 max-h-96 overflow-y-auto">
        {alerts.map((a) => (
          <div
            key={a.alert_id}
            className={`rounded border-l-4 p-3 ${SEVERITY_COLORS[a.severity]}`}
          >
            <div className="flex justify-between">
              <span className="font-mono text-sm">{a.alert_id}</span>
              <span className="text-xs uppercase">{a.severity}</span>
            </div>
            <div className="mt-1 text-sm">
              {a.country_code} / {a.indicator_code} -- Score: {a.anomaly_score.toFixed(3)}
            </div>
          </div>
        ))}
        {alerts.length === 0 && (
          <p className="text-gray-500 text-sm">No active alerts</p>
        )}
      </div>
    </div>
  );
}
