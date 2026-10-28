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

