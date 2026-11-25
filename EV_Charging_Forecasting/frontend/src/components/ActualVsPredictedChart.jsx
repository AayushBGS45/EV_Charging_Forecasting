import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { fetchEvaluationComparison } from '../services/api';
import { Award, Star, Loader2, Calendar, AlertCircle } from 'lucide-react';

export default function ActualVsPredictedChart() {
  const [days, setDays] = useState(7);
  const [selectedOrigin, setSelectedOrigin] = useState('');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const res = await fetchEvaluationComparison({
          originDate: selectedOrigin || undefined,
          days: selectedOrigin ? undefined : days,
        });
        setData(res);
      } catch (err) {
        console.error('Failed to load actual vs predicted data:', err);
        setError(err.message || 'Error loading comparison data.');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, [days, selectedOrigin]);

  function formatAxisTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const month = d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
    const day = String(d.getUTCDate()).padStart(2, '0');
    const hour = String(d.getUTCHours()).padStart(2, '0') + ':00';
    return `${month} ${day} ${hour}`;
  }

  const chartSeries = (data?.series || []).map((pt) => ({
    timestamp: pt.timestamp,
    displayTime: formatAxisTime(pt.timestamp),
    actual: pt.actual_kw,
    sarima: pt.sarima_kw,
    chronos: pt.chronos_kw,
  }));

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const pt = payload[0]?.payload;
      const dateStr = pt?.timestamp ? new Date(pt.timestamp).toUTCString() : label;
      return (
        <div className="bg-surface-container-high border border-outline-variant p-3.5 rounded-xl shadow-2xl text-xs flex flex-col gap-2 z-50 min-w-[240px]">
          <div className="font-mono text-on-surface-variant font-medium border-b border-outline-variant/60 pb-1.5">
            {dateStr}
          </div>
          <div className="flex items-center justify-between gap-3 text-primary font-semibold">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-primary"></span>
              Actual Demand:
            </span>
            <span className="text-on-surface font-headline font-bold text-sm">
              {pt.actual !== null ? `${pt.actual.toFixed(2)} kW` : 'N/A'}
            </span>
          </div>
          <div className="flex items-center justify-between gap-3 text-secondary font-semibold">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-secondary"></span>
              SARIMA Forecast:
            </span>
            <span className="text-on-surface font-headline font-bold text-sm">
              {pt.sarima !== null ? `${pt.sarima.toFixed(2)} kW` : 'N/A'}
            </span>
          </div>
          {pt.chronos !== null && pt.chronos !== undefined && (
            <div className="flex items-center justify-between gap-3 text-tertiary font-semibold">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-tertiary"></span>
                Chronos Forecast:
              </span>
              <span className="text-on-surface font-headline font-bold text-sm">
                {pt.chronos.toFixed(2)} kW
              </span>
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <section className="rounded-xl bg-surface-container-lowest border border-outline-variant p-6 flex flex-col gap-6 relative shadow-lg shadow-black/40">
      {/* Header and Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 pb-4 border-b border-outline-variant/60">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <h3 className="font-headline text-lg font-bold text-on-surface tracking-tight">
              Actual vs. Predicted Demand Trajectory
            </h3>
            <span className="px-2 py-0.5 rounded text-[11px] font-label font-semibold text-primary bg-primary/10 border border-primary/20">
              Holdout Test Set
            </span>
          </div>
          <p className="font-body text-xs text-on-surface-variant">
            Comparing ground truth facility power draw against Day-Ahead forecasts (H=24) from SARIMA and Chronos.
          </p>
        </div>

        {/* Filters and Controls */}
        <div className="flex flex-wrap items-center gap-3 text-xs">
          {/* Day range buttons */}
          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant text-xs font-label">
            <span className="text-outline text-[11px] px-2 font-medium">Range:</span>
            {[3, 7, 14].map((d) => (
              <button
                key={d}
                onClick={() => {
                  setSelectedOrigin('');
                  setDays(d);
                }}
                className={`px-2.5 py-1 rounded font-medium transition-all ${
                  !selectedOrigin && days === d
                    ? 'bg-primary text-on-primary font-bold shadow-xs'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {d}d
              </button>
            ))}
          </div>

          {/* Single origin selector */}
          {data?.available_origin_dates && (
            <div className="flex items-center gap-1.5 bg-surface-container px-2.5 py-1 rounded-lg border border-outline-variant text-xs">
              <Calendar className="w-3.5 h-3.5 text-primary" />
              <select
                value={selectedOrigin}
                onChange={(e) => setSelectedOrigin(e.target.value)}
                className="bg-transparent text-on-surface font-mono text-xs focus:outline-none cursor-pointer"
              >
                <option value="" className="bg-surface-container text-on-surface">
                  -- Specific Day Origin --
                </option>
                {data.available_origin_dates.slice(-30).map((dt) => (
                  <option key={dt} value={dt} className="bg-surface-container text-on-surface">
                    {dt} (H=24)
                  </option>
                ))}
              </select>
            </div>
          )}
        </div>
      </div>

      {/* Metrics Banner for the Selected Slice */}
      {data?.metrics && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-surface-container-low border border-primary/30 flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-bold text-primary flex items-center gap-1">
                <Award className="w-3 h-3 text-primary" />
                SARIMA MAE
              </span>
              <span className="font-headline text-lg font-bold text-on-surface mt-0.5">
                {data.metrics.sarima_mae_kw ?? 'N/A'} kW
              </span>
            </div>
            <span className="text-[11px] font-mono text-outline">
              RMSE: {data.metrics.sarima_rmse_kw ?? 'N/A'} kW
            </span>
          </div>

          <div className="p-3 rounded-lg bg-surface-container-low border border-secondary/30 flex items-center justify-between">
            <div className="flex flex-col">
              <span className="text-[10px] uppercase font-bold text-secondary flex items-center gap-1">
                <Star className="w-3 h-3 text-secondary" />
                Chronos MAE
              </span>
              <span className="font-headline text-lg font-bold text-on-surface mt-0.5">
                {data.metrics.chronos_mae_kw ?? 'N/A'} kW
              </span>
            </div>
            <span className="text-[11px] font-mono text-outline">
              RMSE: {data.metrics.chronos_rmse_kw ?? 'N/A'} kW
            </span>
          </div>

          <div className="p-3 rounded-lg bg-surface-container-low border border-outline-variant/60 flex flex-col justify-center">
            <span className="text-[10px] uppercase text-outline font-semibold">Evaluated Horizon</span>
            <span className="font-headline text-sm font-bold text-on-surface mt-0.5">
              Day-Ahead (H=24 Steps)
            </span>
          </div>

          <div className="p-3 rounded-lg bg-surface-container-low border border-outline-variant/60 flex flex-col justify-center">
            <span className="text-[10px] uppercase text-outline font-semibold">Slice Samples</span>
            <span className="font-mono text-sm font-bold text-on-surface mt-0.5">
              {data.count} hourly intervals
            </span>
          </div>
        </div>
      )}

      {/* Chart Viewport */}
      <div className="relative w-full h-80 sm:h-96 select-none">
        {loading && (
          <div className="absolute inset-0 bg-background/70 backdrop-blur-xs flex items-center justify-center z-20 rounded-lg">
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-surface-container border border-outline-variant text-sm font-medium text-primary">
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
              <span>Loading comparison trajectory...</span>
            </div>
          </div>
        )}

        {error ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-on-surface-variant">
            <AlertCircle className="w-8 h-8 text-error mb-2" />
            <p className="font-semibold text-on-surface">Unable to load comparison data</p>
            <p className="text-xs text-outline mt-1">{error}</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartSeries} margin={{ top: 15, right: 30, left: 10, bottom: 25 }}>
              <CartesianGrid stroke="#242c27" strokeDasharray="3 3" vertical={false} />
              <XAxis
                dataKey="displayTime"
                stroke="#86948a"
                tick={{ fill: '#86948a', fontSize: 10, fontFamily: 'Public Sans' }}
                interval="preserveStartEnd"
                minTickGap={45}
              />
              <YAxis
                stroke="#86948a"
                tick={{ fill: '#86948a', fontSize: 10, fontFamily: 'Public Sans' }}
                unit=" kW"
                domain={[0, 'auto']}
              />
              <Tooltip content={<CustomTooltip />} />
              <Legend
                verticalAlign="top"
                align="right"
                wrapperStyle={{ paddingBottom: '10px', fontSize: '12px' }}
              />

              {/* Actual Demand Line */}
              <Line
                type="monotone"
                dataKey="actual"
                name="Actual Demand (kW)"
                stroke="#4edea3"
                strokeWidth={2.2}
                dot={false}
                activeDot={{ r: 5, fill: '#4edea3' }}
              />

              {/* SARIMA Forecast Line */}
              <Line
                type="monotone"
                dataKey="sarima"
                name="SARIMA Forecast (kW)"
                stroke="#adc6ff"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={false}
                activeDot={{ r: 4, fill: '#adc6ff' }}
              />

              {/* Chronos Forecast Line */}
              <Line
                type="monotone"
                dataKey="chronos"
                name="Chronos Forecast (kW)"
                stroke="#ffb95f"
                strokeWidth={2}
                strokeDasharray="2 2"
                dot={false}
                activeDot={{ r: 4, fill: '#ffb95f' }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>
    </section>
  );
}
