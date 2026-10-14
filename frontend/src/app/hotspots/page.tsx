'use client';
/**
 * Hotspots — where reports of one category are expected next week.
 *
 * Admins and agents only. The forecast is an exponentially weighted moving
 * average of each 1 km cell's weekly report counts (alpha 0.3) — a statistic,
 * not a machine-learning model. Every number shown comes from
 * /api/v1/hotspots; with too little history the page says so rather than
 * drawing anything. The role check below only decides what to render: the API
 * itself answers 403 to citizens, because the map aggregates everyone's
 * report locations.
 */
import { useCallback, useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import {
  CATEGORY_LABELS,
  HotspotForecast,
  apiGetHotspots,
  formatDate,
  getStoredUser,
} from '@/lib/api';

const CATEGORIES = ['POTHOLE', 'FLOOD', 'GARBAGE', 'STREETLIGHT', 'FIRE_HAZARD', 'OTHER'];

const STATUS_TEXT: Record<string, string> = {
  AVAILABLE: 'Enough history for the method as evaluated offline.',
  EXPERIMENTAL: 'Shown with less history than the method was evaluated with.',
  UNAVAILABLE: 'No forecast can be made yet.',
};

function osmLink(lat: number, lon: number): string {
  return `https://www.openstreetmap.org/?mlat=${lat.toFixed(5)}&mlon=${lon.toFixed(5)}#map=15/${lat.toFixed(5)}/${lon.toFixed(5)}`;
}

export default function HotspotsPage() {
  const router = useRouter();
  const [category, setCategory] = useState('POTHOLE');
  const [area, setArea] = useState<string | undefined>(undefined);
  const [data, setData] = useState<HotspotForecast | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      setData(await apiGetHotspots(category, area));
    } catch (err: unknown) {
      setData(null);
      setError(err instanceof Error ? err.message : 'Could not load hotspots.');
    } finally {
      setLoading(false);
    }
  }, [category, area]);

  useEffect(() => {
    const user = getStoredUser();
    if (!user || (user.role !== 'ADMIN' && user.role !== 'AGENT')) {
      router.replace('/login');
      return;
    }
    load();
  }, [router, load]);

  const areas = data?.available_areas ?? [];

  return (
    <>
      <Navbar />
      <main className="container page">
        <div className="page-header">
          <div className="page-header__row">
            <div>
              <h1>Hotspots</h1>
              <p>
                Next week&apos;s expected number of reports per 1 km cell, from a weighted
                moving average of past weeks. This is a simple statistic, not an ML model.
              </p>
            </div>
            <div className="button-row">
              <select
                className="form-select form-select--compact"
                aria-label="Service area"
                value={area ?? data?.area ?? ''}
                onChange={(e) => setArea(e.target.value)}
              >
                {areas.map((a) => (
                  <option key={a} value={a}>
                    {a}
                  </option>
                ))}
              </select>
              <select
                className="form-select form-select--compact"
                aria-label="Category"
                value={category}
                onChange={(e) => setCategory(e.target.value)}
              >
                {CATEGORIES.map((c) => (
                  <option key={c} value={c}>
                    {CATEGORY_LABELS[c] ?? c}
                  </option>
                ))}
              </select>
              <button className="btn btn-outline btn-sm" onClick={load} disabled={loading}>
                {loading && <span className="spinner spinner-sm" />} Refresh
              </button>
            </div>
          </div>
        </div>

        {error && <div className="alert alert-error mb-3">{error}</div>}

        {data && (
          <>
            <div className={`alert ${data.status === 'AVAILABLE' ? 'alert-info' : 'alert-warning'} mb-3`}>
              <p>
                <strong>{data.status}</strong> — {STATUS_TEXT[data.status] ?? ''}
                {data.weeks_of_history > 0 && (
                  <>
                    {' '}
                    {data.weeks_of_history} complete week(s) of history
                    {data.min_history_weeks != null && ` (evaluated with ≥ ${data.min_history_weeks})`}.
                  </>
                )}
              </p>
              {data.reasons.map((r) => (
                <p key={r.code}>
                  {r.message} <span className="prediction__code">({r.code})</span>
                </p>
              ))}
              {data.baseline_comparison && <p>{data.baseline_comparison}</p>}
            </div>

            {data.cells.length === 0 ? (
              <div className="card">
                <p className="muted-note">
                  {data.status === 'UNAVAILABLE'
                    ? 'Nothing to show yet.'
                    : 'No reports of this category in the counted weeks.'}
                </p>
              </div>
            ) : (
              <div className="table-wrapper">
                <table className="user-table">
                  <caption className="sr-only">
                    Expected reports next week by 1 km cell, highest first
                  </caption>
                  <thead>
                    <tr>
                      <th scope="col">#</th>
                      <th scope="col">Cell (centre)</th>
                      <th scope="col">Expected next week</th>
                      <th scope="col">Last week</th>
                      <th scope="col">4-week mean</th>
                      <th scope="col">Reports counted</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.cells.map((c, i) => (
                      <tr key={c.cell_id} className="row-static">
                        <td data-label="#">{i + 1}</td>
                        <td data-label="Cell">
                          <a href={osmLink(c.center_lat, c.center_lon)} target="_blank" rel="noreferrer">
                            {c.center_lat.toFixed(4)}, {c.center_lon.toFixed(4)}
                          </a>
                        </td>
                        <td data-label="Expected next week">
                          {c.predicted_next_week_count.toFixed(1)}
                        </td>
                        <td data-label="Last week">{c.last_week_count}</td>
                        <td data-label="4-week mean">{c.mean_last_4_weeks.toFixed(1)}</td>
                        <td data-label="Reports counted">{c.reports_in_history}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
            <p className="prediction__meta mt-2">
              Weeks run Monday–Sunday (UTC); the current week is not counted.
              {data.first_complete_week &&
                (data.weeks_of_history > 0 ? (
                  <> History from {formatDate(data.first_complete_week)}.</>
                ) : (
                  <> The first complete week starts {formatDate(data.first_complete_week)}.</>
                ))}
              {data.cells_total > data.cells.length &&
                ` Showing ${data.cells.length} of ${data.cells_total} cells.`}
              {data.bundle_version && <> Settings from bundle {data.bundle_version}.</>}
            </p>
          </>
        )}
      </main>
    </>
  );
}
