import React from 'react';
import { ArrowUpRight, Radar, Gauge, CheckCircle2 } from 'lucide-react';

export default function SummaryCards({ stats, forecastData, horizon }) {
  // Compute forecast stats from active predictions
  const predictions = forecastData?.predictions || [];
  const forecastPeak = predictions.length > 0
    ? Math.max(...predictions.map((p) => p.predicted_demand_kw))
    : 0;
  
  const peakPrediction = predictions.find((p) => p.predicted_demand_kw === forecastPeak);
  const peakTimeFormatted = peakPrediction
    ? new Date(peakPrediction.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', timeZone: 'UTC' }) + ' UTC'
    : 'N/A';

  const forecastMean = predictions.length > 0
    ? (predictions.reduce((acc, p) => acc + p.predicted_demand_kw, 0) / predictions.length).toFixed(2)
    : '0.00';

  const historicalPeak = stats?.max_hourly_power_kw?.toFixed(2) || '120.61';
  const historicalMean = stats?.mean_hourly_power_kw?.toFixed(2) || '8.32';

  return (
    <section className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
      {/* CARD 1: Peak Historical Demand (HISTORICAL) */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant/80 hover:border-primary/40 transition-colors duration-200 flex flex-col justify-between relative overflow-hidden group">
        <div className="flex items-start justify-between">
          <div className="flex flex-col">
            <span className="font-label text-[10px] uppercase tracking-wider text-primary font-bold">
              Historical Value
            </span>
            <span className="font-label text-xs uppercase tracking-wider text-on-surface-variant font-semibold mt-0.5">
              Peak Historical Demand
            </span>
          </div>
          <span className="inline-flex items-center gap-0.5 px-2 py-0.5 rounded text-[11px] font-label font-bold text-primary bg-primary/10 border border-primary/20">
            <ArrowUpRight className="w-3 h-3" />
            Max Load
          </span>
        </div>

        <div className="mt-3 flex items-baseline gap-2">
          <span className="font-headline text-3xl font-bold tracking-tight text-on-surface">
            {historicalPeak}
          </span>
          <span className="font-label text-sm text-outline font-semibold">kW</span>
        </div>

        <div className="mt-4 pt-3 border-t border-outline-variant/40 flex items-center justify-between text-xs text-on-surface-variant">
          <span className="truncate">Dataset Maximum (2018–2020)</span>
          <span className="text-outline text-[11px]">Caltech ACN</span>
        </div>
      </div>

      {/* CARD 2: Forecast Peak (PREDICTED) */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-secondary/40 hover:border-secondary transition-colors duration-200 flex flex-col justify-between relative overflow-hidden group shadow-sm">
        <div className="flex items-start justify-between">
          <div className="flex flex-col">
            <span className="font-label text-[10px] uppercase tracking-wider text-secondary font-bold">
              Predicted Value
            </span>
            <span className="font-label text-xs uppercase tracking-wider text-on-surface-variant font-semibold mt-0.5">
              Forecast Peak ({horizon}h)
            </span>
          </div>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-label font-medium text-secondary bg-secondary-container/20 border border-secondary/30">
            <Radar className="w-3 h-3" />
            SARIMA Projection
          </span>
        </div>

        <div className="mt-3 flex items-baseline gap-2">
          <span className="font-headline text-3xl font-bold tracking-tight text-secondary">
            {forecastPeak.toFixed(2)}
          </span>
          <span className="font-label text-sm text-secondary/70 font-semibold">kW</span>
        </div>

        <div className="mt-4 pt-3 border-t border-outline-variant/40 flex items-center justify-between text-xs text-on-surface-variant">
          <span>Projected at {peakTimeFormatted}</span>
          <span className="text-secondary/90 font-medium">Next {horizon}h horizon</span>
        </div>
      </div>

      {/* CARD 3: Average Forecast Demand (PREDICTED) */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant/80 hover:border-tertiary/40 transition-colors duration-200 flex flex-col justify-between">
        <div className="flex items-start justify-between">
          <div className="flex flex-col">
            <span className="font-label text-[10px] uppercase tracking-wider text-tertiary font-bold">
              Predicted Value
            </span>
            <span className="font-label text-xs uppercase tracking-wider text-on-surface-variant font-semibold mt-0.5">
              Avg Forecast Demand
            </span>
          </div>
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[11px] font-label font-medium text-tertiary bg-tertiary-container/15 border border-tertiary/30">
            <Gauge className="w-3 h-3" />
            Mean Load
          </span>
        </div>

        <div className="mt-3 flex items-baseline gap-2">
          <span className="font-headline text-3xl font-bold tracking-tight text-on-surface">
            {forecastMean}
          </span>
          <span className="font-label text-sm text-outline font-semibold">kW</span>
        </div>

        <div className="mt-4 pt-3 border-t border-outline-variant/40 flex items-center justify-between text-xs text-on-surface-variant">
          <span>Hist. Mean: {historicalMean} kW</span>
          <span className="text-outline text-[11px]">Facility Baseline</span>
        </div>
      </div>

      {/* CARD 4: Active Model */}
      <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant/80 hover:border-primary/40 transition-colors duration-200 flex flex-col justify-between">
        <div className="flex items-start justify-between">
          <div className="flex flex-col">
            <span className="font-label text-[10px] uppercase tracking-wider text-outline font-bold">
              Model Metadata
            </span>
            <span className="font-label text-xs uppercase tracking-wider text-on-surface-variant font-semibold mt-0.5">
              Production Forecaster
            </span>
          </div>
          <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-label font-bold text-primary bg-primary/10 border border-primary/20">
            <CheckCircle2 className="w-3 h-3 text-primary" />
            Validated
          </span>
        </div>

        <div className="mt-3 flex items-baseline gap-2">
          <span className="font-headline text-2xl font-bold tracking-tight text-on-surface">
            SARIMA
          </span>
          <span className="font-label text-xs text-primary font-mono font-medium">
            (2,0,1)×(0,1,1)₂₄
          </span>
        </div>

        <div className="mt-4 pt-3 border-t border-outline-variant/40 flex items-center justify-between text-xs text-on-surface-variant">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
            <span>Online Serving</span>
          </div>
          <span className="font-mono text-outline text-[11px]">Unit: kW</span>
        </div>
      </div>
    </section>
  );
}

