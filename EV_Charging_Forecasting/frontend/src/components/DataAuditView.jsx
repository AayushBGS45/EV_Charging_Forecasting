import React from 'react';
import { ShieldCheck, AlertCircle, Database, CheckCircle2, FileText, Info } from 'lucide-react';

export default function DataAuditView({ stats }) {
  const auditPoints = [
    {
      title: 'Provisional 7.7 kW Per-Session Cap',
      status: 'Implemented in v2',
      badgeColor: 'text-primary bg-primary/10 border-primary/20',
      description:
        'A physical cap of 7.7 kW (corresponding to standard 32A @ 240V Level 2 EVSE delivery) was introduced during v2 data cleaning. This successfully suppressed meter-carryover cumulative energy jumps, reducing the erroneous historical peak spike from 375.4 kW down to 120.61 kW.',
      caveat:
        'Note: While empirically sound, station-level electrical ratings remain provisional pending verification against primary hardware spec sheets.',
    },
    {
      title: 'Target Definition: Uniform Hourly Energy Allocation',
      status: 'Target Specification',
      badgeColor: 'text-secondary bg-secondary-container/20 border-secondary/30',
      description:
        'The forecast target column "avg_power_kw" represents hourly facility energy consumption (kWh / 1h = kW) calculated via uniform distribution of energyDelivered over connection duration.',
      caveat:
        'Crucial Caveat: The target does NOT represent instantaneous measured electrical charging power. CC/CV tapering, station curtailment, and intraday throttling are smoothed out by definition.',
    },
    {
      title: 'Strict Causal Temporal Partitioning',
      status: 'Leakage Free',
      badgeColor: 'text-primary bg-primary/10 border-primary/20',
      description:
        'To prevent lookahead bias and data leakage, all models use strict chronological splits: Training (May 2018 – Dec 2019, 14,640 hrs), Validation (Jan – Feb 2020, 1,440 hrs), and Test (Mar – Dec 2020, 6,700 hrs).',
      caveat:
        'Validation set was used strictly for model order selection. Test partition remained sealed until final evaluation.',
    },
  ];

  return (
    <div className="flex flex-col gap-6">
      {/* Overview Banner */}
      <div className="p-6 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-3">
        <div className="flex items-center gap-2.5">
          <ShieldCheck className="w-5 h-5 text-primary" />
          <h2 className="font-headline text-lg font-bold text-on-surface">
            Data Quality & Physical Modeling Audit (Phase 2.1)
          </h2>
        </div>
        <p className="text-xs text-on-surface-variant leading-relaxed max-w-3xl">
          This audit documents data preprocessing standards, corrections applied to the ACN Caltech archive, and methodological boundaries governing time series forecasting models.
        </p>

        {/* Dataset Metadata Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mt-2 pt-3 border-t border-outline-variant/60 text-xs">
          <div>
            <span className="text-outline text-[11px] block">Dataset Archive</span>
            <span className="font-semibold text-on-surface">Caltech ACN v2</span>
          </div>
          <div>
            <span className="text-outline text-[11px] block">Total Sessions</span>
            <span className="font-semibold text-on-surface">{stats?.total_sessions?.toLocaleString() || '26,117'}</span>
          </div>
          <div>
            <span className="text-outline text-[11px] block">Total Energy</span>
            <span className="font-semibold text-primary">{stats?.total_energy_kwh?.toLocaleString() || '258,480'} kWh</span>
          </div>
          <div>
            <span className="text-outline text-[11px] block">Hourly Intervals</span>
            <span className="font-semibold text-on-surface">{stats?.hourly_intervals_recorded?.toLocaleString() || '22,780'} hrs</span>
          </div>
        </div>
      </div>

      {/* Audit Detail Cards */}
      <div className="flex flex-col gap-4">
        {auditPoints.map((pt, idx) => (
          <div
            key={idx}
            className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-2.5"
          >
            <div className="flex items-center justify-between">
              <span className="font-headline text-sm font-bold text-on-surface">{pt.title}</span>
              <span className={`px-2.5 py-0.5 rounded text-[11px] font-label font-bold border ${pt.badgeColor}`}>
                {pt.status}
              </span>
            </div>

            <p className="text-xs text-on-surface-variant leading-relaxed">{pt.description}</p>

            <div className="p-3 rounded-lg bg-surface-container-low border border-outline-variant/40 flex items-start gap-2 text-[11px] text-tertiary">
              <Info className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>{pt.caveat}</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

