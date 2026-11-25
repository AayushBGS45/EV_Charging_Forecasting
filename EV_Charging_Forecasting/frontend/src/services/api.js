/**
 * API Service for EV Charging Demand Forecasting Backend
 */

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = 'API Request failed';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || JSON.stringify(errJson);
    } catch {
      errorDetail = `Status ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export async function fetchForecast(horizon = 24) {
  const res = await fetch(`${API_BASE}/api/forecast?horizon=${horizon}`);
  return handleResponse(res);
}

export async function fetchStats() {
  const res = await fetch(`${API_BASE}/api/stats`);
  return handleResponse(res);
}

export async function fetchHourlyDemand(params = {}) {
  let queryObj = {};
  if (typeof params === 'number') {
    queryObj = { limit: params, recent: true, version: 'v2' };
  } else {
    queryObj = { limit: 168, recent: true, version: 'v2', ...params };
  }

  const query = new URLSearchParams();
  if (queryObj.limit) query.append('limit', queryObj.limit);
  if (queryObj.recent) query.append('recent', 'true');
  if (queryObj.version) query.append('version', queryObj.version);
  if (queryObj.startDate) query.append('start_date', queryObj.startDate);
  if (queryObj.endDate) query.append('end_date', queryObj.endDate);

  const qs = query.toString();
  const url = `${API_BASE}/api/demand/hourly${qs ? `?${qs}` : ''}`;
  const res = await fetch(url);
  return handleResponse(res);
}

export async function fetchDiurnalProfile() {
  const res = await fetch(`${API_BASE}/api/demand/diurnal`);
  return handleResponse(res);
}

export async function fetchStationarity() {
  const res = await fetch(`${API_BASE}/api/demand/stationarity`);
  return handleResponse(res);
}

export async function fetchRecentSessions(limit = 10) {
  const res = await fetch(`${API_BASE}/api/sessions/recent?limit=${limit}`);
  return handleResponse(res);
}

export async function fetchEvaluationComparison({ originDate, days = 7 } = {}) {
  const query = new URLSearchParams();
  if (originDate) query.append('origin_date', originDate);
  if (days) query.append('days', days);
  const qs = query.toString();
  const res = await fetch(`${API_BASE}/api/evaluation/comparison${qs ? `?${qs}` : ''}`);
  return handleResponse(res);
}

export async function fetchEvaluationResiduals({ model = 'sarima', bins = 20, recentDays = 14 } = {}) {
  const query = new URLSearchParams();
  if (model) query.append('model', model);
  if (bins) query.append('bins', bins);
  if (recentDays) query.append('recent_days', recentDays);
  const qs = query.toString();
  const res = await fetch(`${API_BASE}/api/evaluation/residuals${qs ? `?${qs}` : ''}`);
  return handleResponse(res);
}

export async function fetchWeeklyDemand(limit = 104) {
  const res = await fetch(`${API_BASE}/api/demand/weekly?limit=${limit}`);
  return handleResponse(res);
}

export async function fetchCapAuditComparison() {
  const res = await fetch(`${API_BASE}/api/audit/cap-comparison`);
  return handleResponse(res);
}


