import React, { useState } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Area,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
} from 'recharts';
import { Info, AlertCircle, Loader2, Calendar } from 'lucide-react';

export default function ForecastChart({
  historicalData,
  forecastData,
  horizon,
  onSelectHorizon,
  isLoading,
  error,
}) {
  // Configurable historical context window (hours before origin): 24h, 48h, 72h, or 168h
  const [histWindow, setHistWindow] = useState(48);

  const originTimestamp = forecastData?.forecast_origin || '2020-12-05T03:00:00Z';

  // Format date strictly in UTC
  function formatAxisDate(ts) {
    if (!ts) return '';
    const d = new Date(ts);
    const month = d.toLocaleString('en-US', { month: 'short', timeZone: 'UTC' });
    const day = String(d.getUTCDate()).padStart(2, '0');
    const hour = String(d.getUTCHours()).padStart(2, '0') + ':00';
    return `${month} ${day} ${hour}`;
  }

  // Filter historical observations up to forecast origin
  const validHistory = (historicalData || [])
    .filter((item) => new Date(item.timestamp) <= new Date(originTimestamp))
    .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

  // Take the sensible window of recent historical records immediately before origin
  const slicedHistory = validHistory.slice(-histWindow);

  // Build merged chronological chart points
  const chartData = [];

  slicedHistory.forEach((item, idx) => {
    const isAtOrigin = item.timestamp === originTimestamp || idx === slicedHistory.length - 1;
    const val = Number(item.avg_power_kw.toFixed(2));

    if (isAtOrigin) {
      // Origin transition point: connects historical area and initiates dashed forecast line
      chartData.push({
        timestamp: item.timestamp,
        displayTime: formatAxisDate(item.timestamp),
        fullTime: item.timestamp,
        historicalDemand: val,
        forecastDemand: val, // Bridge seamlessly from the origin point
        isOrigin: true,
      });
    } else {
      chartData.push({
        timestamp: item.timestamp,
        displayTime: formatAxisDate(item.timestamp),
        fullTime: item.timestamp,
        historicalDemand: val,
        forecastDemand: null,
        isHistorical: true,
      });
    }
  });

  // Append out-of-sample SARIMA predictions starting at 2020-12-05T04:00:00Z
  (forecastData?.predictions || []).forEach((item) => {
    if (item.timestamp === originTimestamp) return;

    chartData.push({
      timestamp: item.timestamp,
      displayTime: formatAxisDate(item.timestamp),
      fullTime: item.timestamp,
      historicalDemand: null,
      forecastDemand: Number(item.predicted_demand_kw.toFixed(2)),
      isForecast: true,
    });
  });

  // Calculate timeline bounds for header metadata
  const histStartTime = slicedHistory.length > 0 ? formatAxisDate(slicedHistory[0].timestamp) : 'N/A';
  const histEndTime = formatAxisDate(originTimestamp);
  const forecastEndTime =
    forecastData?.predictions?.length > 0
      ? formatAxisDate(forecastData.predictions[forecastData.predictions.length - 1].timestamp)
      : 'N/A';

  const CustomTooltip = ({ active, payload }) => {
    if (active && payload && payload.length) {
      const pt = payload[0]?.payload;
      const dateStr = pt?.fullTime ? new Date(pt.fullTime).toUTCString() : '';
      return (
        <div className="bg-surface-container-high border border-outline-variant p-3.5 rounded-xl shadow-2xl text-xs flex flex-col gap-2 z-50 min-w-[240px]">
          <div className="font-mono text-on-surface-variant font-medium border-b border-outline-variant/60 pb-1.5 flex items-center justify-between">
            <span>{dateStr}</span>
            {pt?.isOrigin && (
              <span className="text-[10px] text-primary bg-primary/10 border border-primary/30 px-1.5 py-0.2 rounded font-bold">
                ORIGIN
              </span>
            )}
            {pt?.isForecast && (
              <span className="text-[10px] text-secondary bg-secondary-container/20 border border-secondary/30 px-1.5 py-0.2 rounded font-medium">
                PREDICTION
              </span>
            )}
            {pt?.isHistorical && (
              <span className="text-[10px] text-primary/80 bg-surface-container px-1.5 py-0.2 rounded">
                OBSERVED
              </span>
            )}
          </div>

          {pt?.historicalDemand !== null && pt?.historicalDemand !== undefined && (
            <div className="flex items-center justify-between gap-3 text-primary font-semibold">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-primary shadow-xs"></span>
                Historical Actual:
              </span>
              <span className="text-on-surface font-headline font-bold text-sm">
                {pt.historicalDemand.toFixed(2)} kW
              </span>
            </div>
          )}

          {pt?.forecastDemand !== null && pt?.forecastDemand !== undefined && !pt?.isOrigin && (
            <div className="flex items-center justify-between gap-3 text-secondary font-semibold">
              <span className="flex items-center gap-1.5">
                <span className="w-2.5 h-2.5 rounded-full bg-secondary shadow-xs"></span>
                SARIMA Forecast:
              </span>
              <span className="text-on-surface font-headline font-bold text-sm">
                {pt.forecastDemand.toFixed(2)} kW
              </span>
            </div>
          )}

          {pt?.isOrigin && (
            <div className="text-[11px] text-primary/90 font-medium italic pt-1 border-t border-outline-variant/40">
              Forecast Origin — Causal predictions project outward from this observation.
            </div>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <section className="rounded-xl bg-surface-container-lowest border border-outline-variant p-6 flex flex-col gap-6 relative shadow-lg shadow-black/40">
      {/* Chart Header & Controls */}
      <div className="flex flex-col xl:flex-row xl:items-center justify-between gap-4 pb-4 border-b border-outline-variant/60">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <h2 className="font-headline text-lg font-bold text-on-surface tracking-tight">
              Hourly Demand & Forecast Timeline
            </h2>
            <span className="px-2 py-0.5 rounded text-[11px] font-label font-semibold text-primary bg-primary/10 border border-primary/20">
              Origin: Dec 5, 2020 03:00 UTC
            </span>
          </div>
          <p className="font-body text-xs text-on-surface-variant flex items-center gap-2 flex-wrap">
            <span>Historical window: <strong className="text-on-surface">{histStartTime}</strong> → <strong className="text-primary">{histEndTime}</strong></span>
            <span>•</span>
            <span>Forecast projection: <strong className="text-secondary">{forecastEndTime}</strong> (+{horizon}h)</span>
          </p>
        </div>

        {/* Legend Swatches, Context Window, and Horizon Selector */}
        <div className="flex flex-wrap items-center gap-4">
          {/* Legend pills */}
          <div className="flex items-center gap-3 text-xs font-label">
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-1 rounded bg-primary shadow-sm shadow-primary/40"></span>
              <span className="text-on-surface font-medium">Historical Actual</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-3.5 h-1 border-t-2 border-dashed border-secondary inline-block"></span>
              <span className="text-secondary font-medium">SARIMA Forecast</span>
            </div>
          </div>

          {/* Historical Window Selector */}
          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant/80 text-xs font-label">
            <span className="text-outline text-[11px] px-2 font-medium">History:</span>
            {[24, 48, 72, 168].map((w) => (
              <button
                key={w}
                onClick={() => setHistWindow(w)}
                className={`px-2.5 py-1 rounded font-medium transition-all duration-150 ${
                  histWindow === w
                    ? 'bg-surface-container-highest text-primary font-bold shadow-xs'
                    : 'text-on-surface-variant hover:text-on-surface'
                }`}
                title={`Show ${w} hours of history prior to origin`}
              >
                {w === 168 ? '7d' : `${w}h`}
              </button>
            ))}
          </div>

          {/* Forecast Horizon Selector */}
          <div className="flex items-center bg-surface-container p-1 rounded-lg border border-outline-variant/80 text-xs font-label">
            <span className="text-outline text-[11px] px-2 font-medium">Forecast:</span>
            {[24, 48, 168].map((h) => {
              const active = horizon === h;
              return (
                <button
                  key={h}
                  onClick={() => onSelectHorizon(h)}
                  disabled={isLoading}
                  className={`px-3 py-1 rounded font-bold transition-all duration-150 ${
                    active
                      ? 'bg-primary text-on-primary shadow-sm'
                      : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container-high'
                  }`}
                >
                  {h}h
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Chart Viewport */}
      <div className="relative w-full h-80 sm:h-96 select-none">
        {isLoading && (
          <div className="absolute inset-0 bg-background/70 backdrop-blur-xs flex items-center justify-center z-20 rounded-lg">
            <div className="flex items-center gap-3 px-4 py-2 rounded-lg bg-surface-container border border-outline-variant text-sm font-medium text-primary">
              <Loader2 className="w-4 h-4 animate-spin text-primary" />
              <span>Generating {horizon}-hour SARIMA forecast...</span>
            </div>
          </div>
        )}

        {error ? (
          <div className="h-full flex flex-col items-center justify-center text-center p-6 text-on-surface-variant">
            <AlertCircle className="w-8 h-8 text-error mb-2" />
            <p className="font-semibold text-on-surface">Unable to load forecast data</p>
            <p className="text-xs text-outline mt-1 max-w-md">{error}</p>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartData} margin={{ top: 25, right: 30, left: 10, bottom: 25 }}>
              <defs>
                <linearGradient id="emeraldGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#4edea3" stopOpacity={0.3} />
                  <stop offset="70%" stopColor="#4edea3" stopOpacity={0.05} />
                  <stop offset="100%" stopColor="#4edea3" stopOpacity={0.0} />
                </linearGradient>
              </defs>

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

              {/* Vertical Reference Line at Forecast Origin */}
              {originTimestamp && (
                <ReferenceLine
                  x={formatAxisDate(originTimestamp)}
                  stroke="#4edea3"
                  strokeDasharray="4 4"
                  strokeWidth={2}
                  label={{
                    value: 'Origin: Dec 5, 03:00 UTC',
                    position: 'top',
                    fill: '#4edea3',
                    fontSize: 10,
                    fontWeight: 700,
                  }}
                />
              )}

              {/* Historical Demand Area + Line */}
              <Area
                type="monotone"
                dataKey="historicalDemand"
                stroke="#4edea3"
                strokeWidth={2}
                fill="url(#emeraldGradient)"
                isAnimationActive={false}
                connectNulls={false}
              />

              {/* SARIMA Forecast Line (High Contrast Dashed Blue) */}
              <Line
                type="monotone"
                dataKey="forecastDemand"
                stroke="#adc6ff"
                strokeWidth={2.5}
                strokeDasharray="5 5"
                dot={{ r: 2, fill: '#adc6ff' }}
                activeDot={{ r: 5, fill: '#adc6ff' }}
                isAnimationActive={false}
                connectNulls={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Footer Notice Bar */}
      <div className="flex items-center justify-between pt-3 border-t border-outline-variant/40 text-xs text-on-surface-variant flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <Info className="w-4 h-4 text-tertiary flex-shrink-0" />
          <span>
            <strong className="text-on-surface">Timeline Continuity:</strong> Showing actual observed demand ({histWindow}h context) up to origin <code className="text-primary font-mono font-semibold">2020-12-05T03:00:00Z</code>, followed by {horizon} consecutive hourly SARIMA predictions starting at <code className="text-secondary font-mono font-semibold">2020-12-05T04:00:00Z</code>.
          </span>
        </div>
        <div className="flex items-center gap-4 text-[11px] font-label text-outline">
          <span>Target: Reconstructed Hourly Energy (kW)</span>
          <span>Caltech ACN Archive</span>
        </div>
      </div>
    </section>
  );
}
