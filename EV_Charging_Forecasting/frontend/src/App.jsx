import React, { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import SummaryCards from './components/SummaryCards';
import ForecastChart from './components/ForecastChart';
import ModelComparison from './components/ModelComparison';
import DemandForecastView from './components/DemandForecastView';
import HistoricalAnalysisView from './components/HistoricalAnalysisView';
import DataAuditView from './components/DataAuditView';
import { fetchForecast, fetchStats, fetchHourlyDemand } from './services/api';
import { AlertTriangle, Server, ExternalLink } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [horizon, setHorizon] = useState(24);
  const [forecastData, setForecastData] = useState(null);
  const [stats, setStats] = useState(null);
  const [historicalData, setHistoricalData] = useState([]);
  const [backendStatus, setBackendStatus] = useState('connecting');
  const [isLoadingForecast, setIsLoadingForecast] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [apiError, setApiError] = useState(null);

  // Load initial dataset stats and historical points leading up to forecast origin
  const loadBaselineData = useCallback(async () => {
    try {
      const [statsRes, hourlyRes] = await Promise.all([
        fetchStats(),
        fetchHourlyDemand({ limit: 168, recent: true, version: 'v2' }),
      ]);
      setStats(statsRes);
      // Ensure chronological ordering
      const sorted = (hourlyRes.data || []).sort(
        (a, b) => new Date(a.timestamp) - new Date(b.timestamp)
      );
      setHistoricalData(sorted);
      setBackendStatus('connected');
    } catch (err) {
      console.warn('Backend baseline data fetch issue:', err.message);
      setBackendStatus('error');
    }
  }, []);

  // Fetch forecast for a specified horizon
  const loadForecast = useCallback(async (h) => {
    try {
      setIsLoadingForecast(true);
      setApiError(null);
      const data = await fetchForecast(h);
      setForecastData(data);
      setBackendStatus('connected');
    } catch (err) {
      console.error('Forecast endpoint error:', err);
      setApiError(err.message || 'Failed to retrieve forecast from FastAPI.');
      setBackendStatus('error');
    } finally {
      setIsLoadingForecast(false);
    }
  }, []);

  // Initial load
  useEffect(() => {
    loadBaselineData();
    loadForecast(24);
  }, [loadBaselineData, loadForecast]);

  // Handle horizon selector change
  const handleSelectHorizon = (newH) => {
    setHorizon(newH);
    loadForecast(newH);
  };

  // Manual refresh handler
  const handleRefresh = async () => {
    setIsRefreshing(true);
    await Promise.all([loadBaselineData(), loadForecast(horizon)]);
    setIsRefreshing(false);
  };

  return (
    <div className="flex min-h-screen bg-background text-on-surface font-body">
      {/* Sidebar Navigation */}
      <Sidebar activeTab={activeTab} onSelectTab={setActiveTab} />

      {/* Main Content Area */}
      <div className="pl-60 flex-1 flex flex-col min-w-0 bg-background">
        {/* Top Header */}
        <Header
          onRefresh={handleRefresh}
          isRefreshing={isRefreshing}
          backendStatus={backendStatus}
        />

        {/* Global Error Banner if Backend is offline */}
        {apiError && (
          <div className="mx-6 mt-4 p-4 rounded-xl bg-red-950/40 border border-red-500/30 flex items-start gap-3 text-xs text-red-200">
            <AlertTriangle className="w-4 h-4 text-red-400 flex-shrink-0 mt-0.5" />
            <div className="flex flex-col gap-1">
              <span className="font-bold text-red-300">FastAPI Connection Warning</span>
              <p>{apiError}</p>
              <p className="text-[11px] text-red-400">
                Ensure backend server is running via{' '}
                <code className="bg-red-900/50 px-1 py-0.5 rounded text-white font-mono">
                  uvicorn backend.main:app --port 8000
                </code>
              </p>
            </div>
          </div>
        )}

        {/* Tab Content Router */}
        <main className="p-6 flex flex-col gap-6 max-w-[1600px] w-full mx-auto">
          {activeTab === 'overview' && (
            <>
              {/* Summary Metrics */}
              <SummaryCards
                stats={stats}
                forecastData={forecastData}
                horizon={horizon}
              />

              {/* Main Interactive Time Series Chart */}
              <ForecastChart
                historicalData={historicalData}
                forecastData={forecastData}
                horizon={horizon}
                onSelectHorizon={handleSelectHorizon}
                isLoading={isLoadingForecast}
                error={apiError}
              />

              {/* Model Comparison Cards */}
              <ModelComparison />
            </>
          )}

          {activeTab === 'forecast' && (
            <DemandForecastView
              forecastData={forecastData}
              horizon={horizon}
              onSelectHorizon={handleSelectHorizon}
              isLoading={isLoadingForecast}
            />
          )}

          {activeTab === 'historical' && <HistoricalAnalysisView />}

          {activeTab === 'performance' && (
            <div className="flex flex-col gap-6">
              <ModelComparison />
            </div>
          )}

          {activeTab === 'audit' && <DataAuditView stats={stats} />}

          {/* System Telemetry Footer Strip */}
          <footer className="mt-2 mb-6 p-4 rounded-xl bg-surface-container-lowest border border-outline-variant/60 flex flex-col md:flex-row items-center justify-between text-xs text-on-surface-variant gap-3">
            <div className="flex items-center gap-3">
              <div className="w-2 h-2 rounded-full bg-primary animate-pulse"></div>
              <span>Caltech Adaptive Charging Network (ACN) API Active</span>
              <span className="text-outline">•</span>
              <span className="text-outline font-mono">Production SARIMA (2,0,1)×(0,1,1)₂₄</span>
            </div>
            <div className="flex items-center gap-6 text-[11px] font-label text-outline">
              <span>Forecast Origin: 2020-12-05 03:00 UTC</span>
              <span className="font-mono">FastAPI /api/forecast</span>
            </div>
          </footer>
        </main>
      </div>
    </div>
  );
}

