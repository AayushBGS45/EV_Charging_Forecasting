import React, { useState } from 'react';
import { Download, Calendar, Zap, AlertTriangle, Clock } from 'lucide-react';

export default function DemandForecastView({ forecastData, horizon, onSelectHorizon, isLoading }) {
  const [filterQuery, setFilterQuery] = useState('');
  const predictions = forecastData?.predictions || [];

  const filteredPredictions = predictions.filter((p) =>
    p.timestamp.toLowerCase().includes(filterQuery.toLowerCase())
  );

  const forecastPeak = predictions.length > 0
    ? Math.max(...predictions.map((p) => p.predicted_demand_kw))
    : 0;

  const peakPoint = predictions.find((p) => p.predicted_demand_kw === forecastPeak);

  function handleExportJSON() {
    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(forecastData, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `sarima_forecast_${horizon}h.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  }

  function handleExportCSV() {
    let csv = 'timestamp,predicted_demand_kw\n';
    predictions.forEach((p) => {
      csv += `${p.timestamp},${p.predicted_demand_kw}\n`;
    });
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.setAttribute('href', url);
    a.setAttribute('download', `sarima_forecast_${horizon}h.csv`);
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Top Controls Bar */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h2 className="font-headline text-lg font-bold text-on-surface">
            Operational Demand Forecast Dispatch
          </h2>
          <p className="font-body text-xs text-on-surface-variant">
            Next {horizon} hourly intervals projecting facility charging demand from origin{' '}
            <span className="font-mono text-primary">{forecastData?.forecast_origin || '2020-12-05T03:00:00Z'}</span>.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          {/* Horizon Toggle */}
          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant text-xs">
            {[24, 48, 168].map((h) => (
              <button
                key={h}
                onClick={() => onSelectHorizon(h)}
                className={`px-3 py-1 rounded font-bold transition-all ${
                  horizon === h
                    ? 'bg-primary text-on-primary shadow-sm'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {h}h
              </button>
            ))}
          </div>

          <button
            onClick={handleExportCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-surface-container-high text-on-surface hover:bg-surface-variant text-xs font-medium border border-outline-variant"
          >
            <Download className="w-3.5 h-3.5" />
            <span>CSV</span>
          </button>

          <button
            onClick={handleExportJSON}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-primary-container text-on-primary text-xs font-semibold hover:brightness-105"
          >
            <Download className="w-3.5 h-3.5" />
            <span>JSON</span>
          </button>
        </div>
      </div>

      {/* Peak Load Advisory */}
      {peakPoint && (
        <div className="p-4 rounded-xl bg-secondary-container/15 border border-secondary/30 flex items-start gap-3">
          <Clock className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
          <div className="flex flex-col gap-1 text-xs text-on-surface">
            <span className="font-headline font-bold text-secondary text-sm">
              Peak Forecast Advisory: {peakPoint.predicted_demand_kw} kW
            </span>
            <p className="text-on-surface-variant">
              The model projects peak power draw of{' '}
              <strong className="text-secondary">{peakPoint.predicted_demand_kw} kW</strong> at{' '}
              <code className="text-on-surface font-mono">{peakPoint.timestamp}</code>. Ensure facility transformer capacity (exceeds threshold) is monitored during this interval.
            </p>
          </div>
        </div>
      )}

      {/* Hourly Predictions Table */}
      <div className="rounded-xl bg-surface-container-lowest border border-outline-variant overflow-hidden">
        <div className="p-4 border-b border-outline-variant/60 flex items-center justify-between">
          <span className="font-headline text-sm font-bold text-on-surface">
            Predicted Demand Intervals ({predictions.length} Hours)
          </span>
          <input
            type="text"
            placeholder="Filter by timestamp..."
            value={filterQuery}
            onChange={(e) => setFilterQuery(e.target.value)}
            className="px-3 py-1 rounded-lg bg-surface-container border border-outline-variant text-xs text-on-surface placeholder:text-outline focus:outline-none focus:border-primary"
          />
        </div>

        <div className="overflow-x-auto max-h-[500px]">
          <table className="w-full text-left text-xs">
            <thead className="bg-surface-container-high text-on-surface-variant sticky top-0 z-10 font-label uppercase text-[10px] tracking-wider">
              <tr>
                <th className="py-2.5 px-4">Step</th>
                <th className="py-2.5 px-4">Prediction Timestamp (UTC)</th>
                <th className="py-2.5 px-4">Predicted Demand (kW)</th>
                <th className="py-2.5 px-4">Model Applied</th>
                <th className="py-2.5 px-4 text-right">Relative Load</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/30 text-on-surface font-mono">
              {filteredPredictions.map((row, idx) => {
                const pct = forecastPeak > 0 ? (row.predicted_demand_kw / forecastPeak) * 100 : 0;
                return (
                  <tr key={row.timestamp} className="hover:bg-surface-container/50 transition-colors">
                    <td className="py-2 px-4 text-outline">+{idx + 1}h</td>
                    <td className="py-2 px-4 font-sans">{new Date(row.timestamp).toUTCString()}</td>
                    <td className="py-2 px-4 font-bold text-primary text-sm">
                      {row.predicted_demand_kw.toFixed(2)}
                    </td>
                    <td className="py-2 px-4 text-on-surface-variant font-sans">
                      SARIMA (2,0,1)×(0,1,1)₂₄
                    </td>
                    <td className="py-2 px-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <div className="w-16 bg-surface-container rounded-full h-1.5 overflow-hidden">
                          <div
                            className="bg-primary h-full rounded-full"
                            style={{ width: `${pct}%` }}
                          ></div>
                        </div>
                        <span className="text-[10px] text-outline w-8">{pct.toFixed(0)}%</span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

