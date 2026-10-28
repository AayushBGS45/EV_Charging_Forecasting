import React from 'react';
import {
  Zap,
  LayoutDashboard,
  TrendingUp,
  History,
  Cpu,
  FileCheck2,
  Settings,
  HelpCircle,
} from 'lucide-react';

export default function Sidebar({ activeTab, onSelectTab }) {
  const navItems = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'forecast', label: 'Demand Forecast', icon: TrendingUp },
    { id: 'historical', label: 'Historical Analysis', icon: History },
    { id: 'performance', label: 'Model Performance', icon: Cpu },
    { id: 'audit', label: 'Data Audit', icon: FileCheck2 },
  ];

  return (
    <aside className="fixed top-0 left-0 h-screen w-60 bg-surface-container-lowest border-r border-outline-variant z-40 flex flex-col justify-between p-4 select-none">
      <div className="flex flex-col gap-6">
        {/* Brand Header */}
        <div className="flex items-center gap-3 px-2 pt-1">
          <div className="w-9 h-9 rounded-lg bg-surface-container-high border border-outline-variant/60 flex items-center justify-center text-primary shadow-sm shadow-primary/10">
            <Zap className="w-5 h-5 text-primary" />
          </div>
          <div className="flex flex-col">
            <span className="font-headline text-base font-bold text-on-surface tracking-tight leading-tight">
              EV Forecast
            </span>
            <span className="font-label text-[11px] text-on-surface-variant font-medium">
              Caltech ACN Facility
            </span>
          </div>
        </div>

        {/* Primary Navigation Links */}
        <nav className="flex flex-col gap-1.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 text-left ${
                  isActive
                    ? 'bg-surface-container-high text-primary border-l-2 border-primary shadow-sm font-semibold'
                    : 'text-on-surface-variant hover:text-on-surface hover:bg-surface-container'
                }`}
              >
                <Icon className={`w-4 h-4 ${isActive ? 'text-primary' : 'text-on-surface-variant'}`} />
                <span>{item.label}</span>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Sidebar Bottom: Secondary Links & Facility Status Badge */}
      <div className="flex flex-col gap-4 pt-4 border-t border-outline-variant/40">
        <div className="flex flex-col gap-0.5 text-xs text-on-surface-variant">
          <div className="flex items-center gap-2 px-3 py-1.5">
            <Settings className="w-3.5 h-3.5" />
            <span>FastAPI Backend 1.0</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-1.5">
            <HelpCircle className="w-3.5 h-3.5" />
            <span>ACN Research Archive</span>
          </div>
        </div>

        {/* Caltech ACN Facility Status Widget */}
        <div className="p-3 rounded-lg bg-surface-container-low border border-outline-variant/60 flex flex-col gap-1.5">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-primary opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-primary"></span>
              </span>
              <span className="font-headline text-xs font-semibold text-on-surface tracking-tight">
                Caltech ACN
              </span>
            </div>
            <span className="font-label text-[10px] text-primary bg-primary/10 border border-primary/20 px-1.5 py-0.5 rounded font-medium">
              READY
            </span>
          </div>
          <p className="font-label text-[11px] text-on-surface-variant flex items-center justify-between">
            <span>54 EVSE Ports</span>
            <span className="text-outline">Pasadena, CA</span>
          </p>
        </div>
      </div>
    </aside>
  );
}

