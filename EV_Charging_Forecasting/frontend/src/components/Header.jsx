import React from 'react';
import { RefreshCw, Database, Server } from 'lucide-react';

export default function Header({ onRefresh, isRefreshing, backendStatus }) {
  return (
    <header className="h-16 px-6 border-b border-outline-variant bg-surface-container-lowest sticky top-0 z-30 flex items-center justify-between">
      {/* Title & Facility Context */}
      <div className="flex items-center gap-4 min-w-0">
        <h1 className="font-headline text-base md:text-lg font-bold text-on-surface tracking-tight truncate">
          Caltech ACN Facility —{' '}
          <span className="font-normal text-on-surface-variant">
            EV Charging Demand Forecasting
          </span>
        </h1>
        <div className="hidden sm:flex items-center gap-2">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-label font-medium bg-surface-container text-primary border border-outline-variant/60">
            <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
            {backendStatus === 'connected' ? 'FastAPI Connected' : 'Connecting...'}
          </span>
          <span className="font-label text-xs text-outline hidden lg:inline">
            Dec 2020 Horizon
          </span>
        </div>
      </div>

      {/* Trailing Controls */}
      <div className="flex items-center gap-3">
        {/* Temporal Scope Pill */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-container text-on-surface-variant text-xs border border-outline-variant/80">
          <Database className="w-3.5 h-3.5 text-primary" />
          <span>Origin: Dec 5, 2020 03:00 UTC</span>
        </div>

        {/* Backend Endpoint Indicator */}
        <div className="hidden lg:flex items-center gap-1.5 text-xs text-outline font-mono">
          <Server className="w-3.5 h-3.5 text-secondary" />
          <span>:8000/api/forecast</span>
        </div>

        {/* Refresh Action Button */}
        <button
          onClick={onRefresh}
          disabled={isRefreshing}
          aria-label="Refresh Data"
          className="p-2 rounded-lg text-on-surface-variant hover:text-on-surface hover:bg-surface-container border border-outline-variant transition-colors duration-150 active:scale-95 disabled:opacity-50"
          title="Refresh forecast and statistics"
        >
          <RefreshCw className={`w-4 h-4 ${isRefreshing ? 'animate-spin text-primary' : ''}`} />
        </button>
      </div>
    </header>
  );
}

