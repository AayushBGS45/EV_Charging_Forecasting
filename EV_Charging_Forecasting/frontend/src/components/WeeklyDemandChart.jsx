import React, { useState, useEffect } from 'react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { fetchWeeklyDemand } from '../services/api';
import { Loader2, Calendar, TrendingDown, Info } from 'lucide-react';

export default function WeeklyDemandChart() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadWeekly() {
      try {
        setLoading(true);
        const res = await fetchWeeklyDemand(104);
        setData(res);
      } catch (err) {
        console.error('Failed to load weekly trends:', err);
      } finally {
        setLoading(false);
      }
    }
    loadWeekly();
  }, []);

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center gap-3 text-primary text-xs">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span>Loading weekly demand trends...</span>
      </div>
    );
  }

  const weeklyList = (data?.data || []).map((row) => ({
    week_start: row.week_start,
    week_label: row.week_label,
    total_energy_kwh: row.total_energy_kwh,
    mean_power_kw: row.mean_power_kw,
    peak_power_kw: row.peak_power_kw,
  }));

  const regimes = data?.regime_averages_kwh;

  return (
    <section className="p-6 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-5">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 border-b border-outline-variant/60 pb-3">
        <div>
          <h3 className="font-headline text-base font-bold text-on-surface">
            Weekly Energy Consumption & Peak Load Trends (2018–2020)
          </h3>
          <p className="font-body text-xs text-on-surface-variant">
            Macro-level weekly aggregated demand highlighting structural shifts, academic seasons, and external shocks.
          </p>
        </div>
        <span className="text-xs font-label text-outline">
          Caltech ACN Facility — {weeklyList.length} Weeks
        </span>
      </div>

      {/* Structural Regime Badges */}
      {regimes && (
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-surface-container-low border border-primary/30 flex flex-col justify-between">
            <span className="text-[10px] uppercase font-bold text-primary">Pre-COVID Baseline (2018–Feb 2020)</span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-lg font-bold text-on-surface">
                {regimes.pre_covid_weekly_kwh.toLocaleString()}
              </span>
              <span className="text-[10px] text-outline">kWh / wk</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Normal academic commuting cycle</span>
          </div>

          <div className="p-3 rounded-lg bg-surface-container-low border border-red-500/30 flex flex-col justify-between">
            <span className="text-[10px] uppercase font-bold text-red-400">Lockdown Shock (Mar–May 2020)</span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-lg font-bold text-red-300">
                {regimes.lockdown_weekly_kwh.toLocaleString()}
              </span>
              <span className="text-[10px] text-outline">kWh / wk</span>
            </div>
            <span className="text-[10px] text-red-400/90 mt-1">-58.6% drop during campus closure</span>
          </div>

          <div className="p-3 rounded-lg bg-surface-container-low border border-secondary/30 flex flex-col justify-between">
            <span className="text-[10px] uppercase font-bold text-secondary">Partial Recovery (Jun–Dec 2020)</span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="font-headline text-lg font-bold text-on-surface">
                {regimes.recovery_weekly_kwh.toLocaleString()}
              </span>
              <span className="text-[10px] text-outline">kWh / wk</span>
            </div>
            <span className="text-[10px] text-on-surface-variant mt-1">Gradual rebound to ~65% of normal</span>
          </div>
        </div>
      )}

      {/* Chart Viewport */}
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={weeklyList} margin={{ top: 10, right: 30, left: 10, bottom: 20 }}>
            <CartesianGrid stroke="#242c27" strokeDasharray="3 3" vertical={false} />
            <XAxis
              dataKey="week_label"
              stroke="#86948a"
              tick={{ fill: '#86948a', fontSize: 10, fontFamily: 'Public Sans' }}
              interval="preserveStartEnd"
              minTickGap={45}
            />
            <YAxis
              yAxisId="left"
              stroke="#4edea3"
              tick={{ fill: '#4edea3', fontSize: 10, fontFamily: 'Public Sans' }}
              unit=" kWh"
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#adc6ff"
              tick={{ fill: '#adc6ff', fontSize: 10, fontFamily: 'Public Sans' }}
              unit=" kW"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#242c27',
                borderColor: '#3c4a42',
                borderRadius: '0.5rem',
                fontSize: '12px',
              }}
              formatter={(val, name) => {
                if (name.includes('Energy')) return [`${val.toLocaleString()} kWh`, name];
                return [`${val} kW`, name];
              }}
            />
            <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: '10px', fontSize: '11px' }} />

            {/* Weekly Total Energy Bars */}
            <Bar
              yAxisId="left"
              dataKey="total_energy_kwh"
              name="Weekly Energy (kWh)"
              fill="#10b981"
              opacity={0.75}
              radius={[3, 3, 0, 0]}
            />

            {/* Weekly Peak Demand Line */}
            <Line
              yAxisId="right"
              type="monotone"
              dataKey="peak_power_kw"
              name="Weekly Peak Power (kW)"
              stroke="#adc6ff"
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4 }}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
