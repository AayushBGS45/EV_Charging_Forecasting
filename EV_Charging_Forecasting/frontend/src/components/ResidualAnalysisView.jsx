import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { fetchEvaluationResiduals } from '../services/api';
import { Loader2, AlertCircle, BarChart3, TrendingUp, Info } from 'lucide-react';

export default function ResidualAnalysisView() {
  const [model, setModel] = useState('sarima');
  const [bins, setBins] = useState(20);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function loadResiduals() {
      try {
        setLoading(true);
        setError(null);
        const res = await fetchEvaluationResiduals({ model, bins, recentDays: 14 });
        setData(res);
      } catch (err) {
        console.error('Failed to load residuals:', err);
        setError(err.message || 'Error loading residual data.');
      } finally {
        setLoading(false);
      }
    }
    loadResiduals();
  }, [model, bins]);

  function formatTime(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const month = d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
    const day = String(d.getUTCDate()).padStart(2, '0');
    const hour = String(d.getUTCHours()).padStart(2, '0') + ':00';
    return `${month} ${day} ${hour}`;
  }

  const timeSeriesData = (data?.recent_series || []).map((pt) => ({
    timestamp: pt.timestamp,
    displayTime: formatTime(pt.timestamp),
    residual: pt.residual_kw,
    actual: pt.actual_kw,
    pred: pt.pred_kw,
  }));

  const stats = data?.statistics;

  return (
    <div className="flex flex-col gap-6">
      {/* Top Banner and Controls */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <h3 className="font-headline text-base font-bold text-on-surface">
            Model Residual & Error Distribution Analysis
          </h3>
          <p className="font-body text-xs text-on-surface-variant">
            Residuals are computed as <code className="text-primary font-mono font-semibold">e_t = Actual - Predicted</code> on 6,696 holdout test intervals.
          </p>
        </div>

        {/* Model Toggle and Bins */}
        <div className="flex items-center gap-3 text-xs">
          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant font-label">
            <span className="text-outline text-[11px] px-2 font-medium">Model:</span>
            {['sarima', 'chronos'].map((m) => (
              <button
                key={m}
                onClick={() => setModel(m)}
                className={`px-3 py-1 rounded font-bold uppercase transition-all ${
                  model === m
                    ? 'bg-primary text-on-primary shadow-xs'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {m}
              </button>
            ))}
          </div>

          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant font-label">
            <span className="text-outline text-[11px] px-2 font-medium">Bins:</span>
            {[15, 20, 30].map((b) => (
              <button
                key={b}
                onClick={() => setBins(b)}
                className={`px-2 py-1 rounded font-medium transition-all ${
                  bins === b
                    ? 'bg-surface-container-highest text-primary font-bold'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
              >
                {b}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Error Statistics Cards */}
      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-3 xl:grid-cols-6 gap-3 text-xs">
          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Mean Bias Error (ME)
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className={`font-headline text-xl font-bold ${Math.abs(stats.mean_error_kw) < 1.0 ? 'text-primary' : 'text-tertiary'}`}>
                {stats.mean_error_kw > 0 ? `+${stats.mean_error_kw}` : stats.mean_error_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Near-zero indicates minimal bias</span>
          </div>

          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Mean Absolute Error
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-xl font-bold text-on-surface">
                {stats.mae_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Average magnitude of error</span>
          </div>

          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Root Mean Squared
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-xl font-bold text-on-surface">
                {stats.rmse_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Penalizes large deviations</span>
          </div>

          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Std Dev (σ)
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-xl font-bold text-on-surface">
                {stats.std_error_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Residual dispersion spread</span>
          </div>

          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Max Overprediction
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-xl font-bold text-secondary">
                {stats.min_error_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Min residual (Pred &gt; Act)</span>
          </div>

          <div className="p-3 rounded-xl bg-surface-container-lowest border border-outline-variant/80 flex flex-col justify-between">
            <span className="text-[10px] uppercase text-outline font-semibold tracking-wider">
              Max Underprediction
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-xl font-bold text-tertiary">
                +{stats.max_error_kw}
              </span>
              <span className="text-[10px] text-outline">kW</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Max residual (Act &gt; Pred)</span>
          </div>
        </div>
      )}

      {/* Residuals Over Time Chart */}
      <section className="p-6 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-outline-variant/60 pb-3">
          <div>
            <h4 className="font-headline text-base font-bold text-on-surface">
              Residual Error Over Time (e_t)
            </h4>
            <p className="font-body text-xs text-on-surface-variant">
              Hourly deviations across a 14-day test slice. Fluctuations around zero show good model centering.
            </p>
          </div>
          <span className="text-xs font-mono text-outline">Zero Reference Line</span>
        </div>

        <div className="h-64 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center gap-2 text-primary text-xs">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading time series...</span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={timeSeriesData} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
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
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#242c27',
                    borderColor: '#3c4a42',
                    borderRadius: '0.5rem',
                    fontSize: '12px',
                  }}
                  formatter={(val) => [`${Number(val).toFixed(2)} kW`, 'Residual (e_t)']}
                />
                <ReferenceLine y={0} stroke="#4edea3" strokeWidth={1.5} strokeDasharray="3 3" />
                <Line
                  type="monotone"
                  dataKey="residual"
                  name="Residual Error (kW)"
                  stroke="#adc6ff"
                  strokeWidth={1.8}
                  dot={{ r: 1.5, fill: '#adc6ff' }}
                  activeDot={{ r: 4 }}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>

      {/* Residual Distribution Histogram */}
      <section className="p-6 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-4">
        <div className="flex items-center justify-between border-b border-outline-variant/60 pb-3">
          <div>
            <h4 className="font-headline text-base font-bold text-on-surface">
              Residual Error Distribution Histogram ({bins} Bins)
            </h4>
            <p className="font-body text-xs text-on-surface-variant">
              Frequencies of prediction residuals across all 6,696 test samples. A symmetric bell shape centered at zero confirms balanced forecasting.
            </p>
          </div>
          <span className="text-xs font-label text-outline">N = 6,696 Samples</span>
        </div>

        <div className="h-64 w-full">
          {loading ? (
            <div className="h-full flex items-center justify-center gap-2 text-primary text-xs">
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Loading histogram...</span>
            </div>
          ) : (
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={data?.histogram || []} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
                <CartesianGrid stroke="#242c27" strokeDasharray="3 3" vertical={false} />
                <XAxis
                  dataKey="bin_center"
                  stroke="#86948a"
                  tick={{ fill: '#86948a', fontSize: 10, fontFamily: 'Public Sans' }}
                  unit=" kW"
                />
                <YAxis
                  stroke="#86948a"
                  tick={{ fill: '#86948a', fontSize: 10, fontFamily: 'Public Sans' }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#242c27',
                    borderColor: '#3c4a42',
                    borderRadius: '0.5rem',
                    fontSize: '12px',
                  }}
                  formatter={(val, name, item) => [
                    `${val} intervals (${item.payload.percentage}%)`,
                    `Range: ${item.payload.bin_range}`,
                  ]}
                />
                <Bar dataKey="count" name="Frequency" fill="#10b981" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>
    </div>
  );
}
