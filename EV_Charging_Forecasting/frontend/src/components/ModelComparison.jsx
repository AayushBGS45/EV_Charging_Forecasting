import React from 'react';
import { Award, Star, Cpu, Activity, Info } from 'lucide-react';

export default function ModelComparison() {
  const models = [
    {
      id: 'sarima',
      name: 'SARIMA',
      specification: '(2,0,1) × (0,1,1)₂₄',
      badge: 'Production Model',
      badgeColor: 'text-primary bg-primary/10 border-primary/30',
      description: 'Seasonal ARIMA with 24-hour diurnal lag difference and MA Kalman state updating.',
      h24: {
        mae: '5.91',
        rmse: '8.51',
        smape: '58.73%',
        mape: '66.40%',
      },
      h1: {
        mae: '3.48',
        rmse: '5.11',
      },
      latency: '~2 ms (cached)',
      evalSamples: '6,696 test origins',
      highlightBorder: 'border-2 border-primary/50 shadow-md',
      tag: 'Classical Parametric',
    },
    {
      id: 'chronos',
      name: 'Chronos',
      specification: 'chronos-t5-tiny (ctx=504)',
      badge: 'Best MAE',
      badgeColor: 'text-secondary bg-secondary-container/20 border-secondary/30',
      description: 'Zero-shot pretrained Transformer foundation model for probabilistic time series forecasting.',
      h24: {
        mae: '5.87',
        rmse: '8.72',
        smape: '61.20%',
        mape: '58.46%',
      },
      h1: {
        mae: '3.33',
        rmse: '5.10',
      },
      latency: '~420 ms (PyTorch)',
      evalSamples: '6,696 test origins',
      highlightBorder: 'border border-secondary/40',
      tag: 'Foundation Model',
    },
    {
      id: 'lightgbm',
      name: 'LightGBM',
      specification: 'LGBM_L2_lr05_nl31',
      badge: 'ML Baseline',
      badgeColor: 'text-tertiary bg-tertiary-container/15 border-tertiary/30',
      description: 'Gradient Boosted Decision Trees trained on causal calendar, rolling statistics, and lag features (lag_168 dominant).',
      h24: {
        mae: '6.68',
        rmse: '9.28',
        smape: '60.29%',
        mape: '78.81%',
      },
      h1: {
        mae: '3.42',
        rmse: '5.14',
      },
      latency: '~8 ms (Booster)',
      evalSamples: '6,696 test origins',
      highlightBorder: 'border border-outline-variant/80',
      tag: 'Tree Ensemble',
    },
  ];

  return (
    <section className="flex flex-col gap-4">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <h3 className="font-headline text-base font-bold text-on-surface tracking-tight">
            Phase 3 Model Benchmark Comparison
          </h3>
          <p className="font-body text-xs text-on-surface-variant">
            Verified Day-Ahead (H=24) and One-Step (H=1) evaluation metrics tested on 6,696–6,700 holdout observations (March–December 2020).
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs font-label text-outline">
          <Activity className="w-3.5 h-3.5 text-primary" />
          <span>Holdout Test Split</span>
        </div>
      </div>

      {/* 3 Comparison Cards Side-by-Side */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {models.map((model) => (
          <div
            key={model.id}
            className={`p-5 rounded-xl bg-surface-container-lowest relative overflow-hidden flex flex-col justify-between ${model.highlightBorder}`}
          >
            <div className="flex flex-col gap-3">
              {/* Card Header */}
              <div className="flex items-center justify-between">
                <div className="flex flex-col">
                  <span className="font-headline text-lg font-bold text-on-surface">
                    {model.name}
                  </span>
                  <span className="font-mono text-xs text-outline">{model.specification}</span>
                </div>
                <span
                  className={`px-2.5 py-0.5 rounded text-[11px] font-label font-bold border flex items-center gap-1 ${model.badgeColor}`}
                >
                  {model.id === 'sarima' && <Award className="w-3 h-3 text-primary" />}
                  {model.id === 'chronos' && <Star className="w-3 h-3 text-secondary" />}
                  {model.id === 'lightgbm' && <Cpu className="w-3 h-3 text-tertiary" />}
                  {model.badge}
                </span>
              </div>

              <p className="font-label text-xs text-on-surface-variant leading-relaxed">
                {model.description}
              </p>

              {/* Day-Ahead Metrics Grid (H=24) */}
              <div className="pt-2">
                <span className="font-label text-[10px] uppercase text-outline font-bold tracking-wider block mb-1.5">
                  Day-Ahead (H=24) Performance
                </span>
                <div className="grid grid-cols-2 gap-2">
                  <div className="p-2.5 rounded-lg bg-surface-container-low border border-outline-variant/60">
                    <span className="font-label text-[10px] uppercase text-outline font-semibold block">
                      MAE
                    </span>
                    <div className="flex items-baseline gap-1 mt-0.5">
                      <span className="font-headline text-xl font-bold text-on-surface">
                        {model.h24.mae}
                      </span>
                      <span className="font-label text-[10px] text-outline">kW</span>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-surface-container-low border border-outline-variant/60">
                    <span className="font-label text-[10px] uppercase text-outline font-semibold block">
                      RMSE
                    </span>
                    <div className="flex items-baseline gap-1 mt-0.5">
                      <span className="font-headline text-xl font-bold text-on-surface">
                        {model.h24.rmse}
                      </span>
                      <span className="font-label text-[10px] text-outline">kW</span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Secondary Metrics */}
              <div className="grid grid-cols-2 gap-2 pt-1 text-[11px] text-on-surface-variant">
                <div className="flex items-center justify-between px-2 py-1 rounded bg-surface-container-low/60">
                  <span className="text-outline">sMAPE:</span>
                  <span className="font-mono font-medium">{model.h24.smape}</span>
                </div>
                <div className="flex items-center justify-between px-2 py-1 rounded bg-surface-container-low/60">
                  <span className="text-outline">1-Step MAE:</span>
                  <span className="font-mono font-medium text-primary">{model.h1.mae} kW</span>
                </div>
              </div>
            </div>

            {/* Card Footer Details */}
            <div className="mt-4 pt-3 border-t border-outline-variant/40 flex items-center justify-between text-xs text-on-surface-variant font-mono">
              <span className="text-outline">{model.tag}</span>
              <span className="font-medium text-on-surface">{model.latency}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Methodological Context Note */}
      <div className="p-3 rounded-lg bg-surface-container-low border border-outline-variant/40 text-xs text-on-surface-variant flex items-start gap-2.5">
        <Info className="w-4 h-4 text-primary flex-shrink-0 mt-0.5" />
        <div className="flex flex-col gap-1">
          <p>
            <strong className="text-on-surface">Target Definition & Provisional Cap:</strong> Demand target (<code className="text-primary font-mono">avg_power_kw</code>) represents hourly facility energy consumption reconstructed through uniform energy distribution across charging sessions. A provisional 7.7 kW per-session cap suppresses meter-carryover anomalies.
          </p>
        </div>
      </div>
    </section>
  );
}

