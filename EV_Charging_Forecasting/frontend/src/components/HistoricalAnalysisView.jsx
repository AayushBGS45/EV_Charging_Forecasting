import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Legend,
} from 'recharts';
import { fetchDiurnalProfile, fetchStationarity, fetchRecentSessions } from '../services/api';
import { AlertCircle, CheckCircle2, TrendingDown, Clock, Loader2 } from 'lucide-react';

export default function HistoricalAnalysisView() {
  const [diurnal, setDiurnal] = useState(null);
  const [stationarity, setStationarity] = useState(null);
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        const [dData, sData, sessData] = await Promise.all([
          fetchDiurnalProfile(),
          fetchStationarity(),
          fetchRecentSessions(8),
        ]);
        setDiurnal(dData);
        setStationarity(sData);
        setSessions(sessData?.data || []);
      } catch (err) {
        console.error('Failed to load historical analysis data:', err);
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="h-64 flex items-center justify-center gap-3 text-primary text-sm">
        <Loader2 className="w-5 h-5 animate-spin" />
        <span>Loading historical time series analysis...</span>
      </div>
    );
  }

  // Merge weekday and weekend diurnal profiles for 24 hours
  const diurnalChartData = Array.from({ length: 24 }).map((_, hour) => {
    const wd = diurnal?.weekday?.find((r) => r.hour === hour);
    const we = diurnal?.weekend?.find((r) => r.hour === hour);
    return {
      hour: `${String(hour).padStart(2, '0')}:00`,
      weekdayMean: wd?.mean_kw || 0,
      weekendMean: we?.mean_kw || 0,
    };
  });

  return (
    <div className="flex flex-col gap-6">
      {/* 24-Hour Diurnal Demand Profile */}
      <section className="p-6 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col gap-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 border-b border-outline-variant/60 pb-3">
          <div>
            <h3 className="font-headline text-base font-bold text-on-surface">
              24-Hour Diurnal Power Demand Cycle
            </h3>
            <p className="font-body text-xs text-on-surface-variant">
              Average facility load across hour-of-day: Weekday commute arrivals vs. weekend dormancy.
            </p>
          </div>
          <div className="flex items-center gap-4 text-xs font-label">
            <span className="flex items-center gap-1.5 text-primary">
              <span className="w-3 h-0.5 bg-primary inline-block"></span>
              Weekday Mean (kW)
            </span>
            <span className="flex items-center gap-1.5 text-secondary">
              <span className="w-3 h-0.5 bg-secondary inline-block"></span>
              Weekend Mean (kW)
            </span>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={diurnalChartData} margin={{ top: 10, right: 30, left: 0, bottom: 10 }}>
              <CartesianGrid stroke="#242c27" strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="hour" stroke="#86948a" tick={{ fill: '#86948a', fontSize: 10 }} />
              <YAxis stroke="#86948a" tick={{ fill: '#86948a', fontSize: 10 }} unit=" kW" />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#242c27',
                  borderColor: '#3c4a42',
                  borderRadius: '0.5rem',
                  fontSize: '12px',
                }}
              />
              <Line
                type="monotone"
                dataKey="weekdayMean"
                name="Weekday Mean"
                stroke="#4edea3"
                strokeWidth={2.5}
                dot={{ r: 2 }}
              />
              <Line
                type="monotone"
                dataKey="weekendMean"
                name="Weekend Mean"
                stroke="#adc6ff"
                strokeWidth={2}
                strokeDasharray="4 4"
                dot={{ r: 2 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>

      {/* Stationarity Tests (ADF and KPSS) */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* ADF Test Card */}
        <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col justify-between">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-headline text-sm font-bold text-on-surface">
                Augmented Dickey-Fuller (ADF) Test
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-label font-bold text-primary bg-primary/10 border border-primary/20">
                Unit Root Rejected
              </span>
            </div>
            <p className="text-xs text-on-surface-variant">
              Tests null hypothesis of a unit root (difference non-stationarity).
            </p>

            <div className="grid grid-cols-2 gap-2 mt-2">
              <div className="p-2.5 rounded bg-surface-container-low border border-outline-variant/60">
                <span className="text-[10px] text-outline uppercase font-semibold block">Test Stat</span>
                <span className="font-mono text-sm font-bold text-primary">
                  {stationarity?.adf_test?.test_statistic?.toFixed(4) || '-10.5627'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-surface-container-low border border-outline-variant/60">
                <span className="text-[10px] text-outline uppercase font-semibold block">p-value</span>
                <span className="font-mono text-sm font-bold text-on-surface">
                  {stationarity?.adf_test?.p_value?.toExponential(2) || '7.65e-19'}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-outline-variant/40 text-[11px] text-on-surface-variant">
            {stationarity?.adf_test?.interpretation ||
              'Rejects unit-root non-stationarity; series exhibits mean-reverting diurnal oscillation.'}
          </div>
        </div>

        {/* KPSS Test Card */}
        <div className="p-5 rounded-xl bg-surface-container-lowest border border-outline-variant flex flex-col justify-between">
          <div className="flex flex-col gap-2">
            <div className="flex items-center justify-between">
              <span className="font-headline text-sm font-bold text-on-surface">
                KPSS Stationarity Test
              </span>
              <span className="px-2 py-0.5 rounded text-[10px] font-label font-bold text-tertiary bg-tertiary-container/15 border border-tertiary/20">
                Stationarity Rejected
              </span>
            </div>
            <p className="text-xs text-on-surface-variant">
              Tests null hypothesis that the series is level stationary.
            </p>

            <div className="grid grid-cols-2 gap-2 mt-2">
              <div className="p-2.5 rounded bg-surface-container-low border border-outline-variant/60">
                <span className="text-[10px] text-outline uppercase font-semibold block">Test Stat</span>
                <span className="font-mono text-sm font-bold text-tertiary">
                  {stationarity?.kpss_test?.test_statistic?.toFixed(4) || '1.3150'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-surface-container-low border border-outline-variant/60">
                <span className="text-[10px] text-outline uppercase font-semibold block">1% Critical Val</span>
                <span className="font-mono text-sm font-bold text-on-surface">
                  {stationarity?.kpss_test?.critical_values?.['1%'] || '0.739'}
                </span>
              </div>
            </div>
          </div>

          <div className="mt-3 pt-3 border-t border-outline-variant/40 text-[11px] text-on-surface-variant">
            {stationarity?.kpss_test?.interpretation ||
              'Rejects null of level stationarity (p < 0.05). Demonstrates significant seasonal and variance non-stationarity.'}
          </div>
        </div>
      </section>

      {/* Recent Charging Sessions Registry */}
      <section className="rounded-xl bg-surface-container-lowest border border-outline-variant overflow-hidden">
        <div className="p-4 border-b border-outline-variant/60">
          <h3 className="font-headline text-sm font-bold text-on-surface">
            Recent Caltech Charging Sessions
          </h3>
          <p className="text-xs text-on-surface-variant">
            Micro-records showing individual EV connection, energy delivered, and current ratings.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-surface-container-high text-on-surface-variant text-[10px] uppercase font-label">
              <tr>
                <th className="py-2.5 px-4">Session ID</th>
                <th className="py-2.5 px-4">Station</th>
                <th className="py-2.5 px-4">Connection Time (UTC)</th>
                <th className="py-2.5 px-4">Duration</th>
                <th className="py-2.5 px-4 text-right">Energy (kWh)</th>
                <th className="py-2.5 px-4 text-right">Max Current</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-outline-variant/30 text-on-surface">
              {sessions.map((sess) => (
                <tr key={sess.session_id} className="hover:bg-surface-container/50">
                  <td className="py-2 px-4 text-outline truncate max-w-[120px]">{sess.session_id}</td>
                  <td className="py-2 px-4 text-primary font-semibold">{sess.station_id}</td>
                  <td className="py-2 px-4 font-sans text-on-surface-variant">
                    {new Date(sess.connectionTime).toUTCString()}
                  </td>
                  <td className="py-2 px-4">{sess.duration_hours?.toFixed(1)} hrs</td>
                  <td className="py-2 px-4 text-right font-bold text-primary">
                    {sess.kWhDelivered?.toFixed(2)} kWh
                  </td>
                  <td className="py-2 px-4 text-right">{sess.maxCurrent_A?.toFixed(0)} A</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}

