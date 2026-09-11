'use client';
import { useState, useEffect, useCallback, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { Navbar } from '@/components/Navbar';
import { IncidentCard } from '@/components/IncidentCard';
import { LocationCapture } from '@/components/LocationCapture';
import { ImageUploadField } from '@/components/ImageUploadField';
import { IncidentImages } from '@/components/IncidentImages';
import { WorkflowTimeline } from '@/components/WorkflowTimeline';
import {
  DetailHeader,
  IncidentMetadata,
  PrioritySummary,
  SlaSummary,
} from '@/components/IncidentDetailParts';
import {
  CATEGORY_LABELS,
  Incident,
  apiCreateIncident,
  apiGetIncidents,
  getStoredUser,
} from '@/lib/api';

const CATEGORIES = ['POTHOLE', 'FLOOD', 'FIRE_HAZARD', 'GARBAGE', 'STREETLIGHT', 'OTHER'];

export default function DashboardPage() {
  const router = useRouter();
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [loadingList, setLoadingList] = useState(true);
  const [listError, setListError] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [selectedId, setSelectedId] = useState<number | null>(null);

  // Form state — note there is no severity/priority field: the system decides.
  const [title, setTitle] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('POTHOLE');
  const [location, setLocation] = useState<{ latitude: number; longitude: number } | null>(null);
  const [images, setImages] = useState<File[]>([]);
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');

  const fetchIncidents = useCallback(async () => {
    setLoadingList(true);
    setListError('');
    try {
      setIncidents(await apiGetIncidents());
    } catch (err: unknown) {
      setListError(err instanceof Error ? err.message : 'Could not load your reports.');
    } finally {
      setLoadingList(false);
    }
  }, []);

  useEffect(() => {
    const user = getStoredUser();
    if (!user || user.role !== 'USER') {
      router.replace('/login');
      return;
    }
    fetchIncidents();
  }, [router, fetchIncidents]);

  const resetForm = () => {
    setTitle('');
    setDescription('');
    setCategory('POTHOLE');
    setLocation(null);
    setImages([]);
    setFormError('');
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!location) {
      setFormError('Please capture or enter the incident location first.');
      return;
    }
    setFormError('');
    setSubmitting(true);
    try {
      const inc = await apiCreateIncident(
        {
          title,
          description,
          category,
          latitude: location.latitude,
          longitude: location.longitude,
        },
        images,
      );
      setIncidents((prev) => [inc, ...prev]);
      setShowForm(false);
      resetForm();
      setSelectedId(inc.id);
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Submission failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const selected = incidents.find((i) => i.id === selectedId) ?? null;
  const panelOpen = showForm || Boolean(selected);

  return (
    <>
      <Navbar />
      <main className="container page">
        <div className="page-header">
          <div className="page-header__row">
            <div>
              <h1>My reports</h1>
              <p>
                Track the incidents you have reported to the city. Priority and
                response deadlines are assigned automatically.
              </p>
            </div>
            <button
              id="btn-new-incident"
              className="btn btn-primary"
              onClick={() => {
                setShowForm(true);
                setSelectedId(null);
              }}
            >
              <span aria-hidden="true">＋</span> Report incident
            </button>
          </div>
        </div>

        <div className={`workspace${panelOpen ? ' workspace--split' : ''}`}>
          <div className="workspace__main">
            {loadingList ? (
              <div className="loading-block" role="status">
                <span className="spinner" />
                <p>Loading your reports…</p>
              </div>
            ) : listError ? (
              <div className="alert alert-error">
                <p>{listError}</p>
                <button className="btn btn-sm btn-outline mt-1" onClick={fetchIncidents}>
                  Try again
                </button>
              </div>
            ) : incidents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-state__icon" aria-hidden="true">📋</div>
                <h3>No reports yet</h3>
                <p>
                  Spotted a pothole, a flood or a broken streetlight? Report it
                  and the city will triage it for you.
                </p>
                <button
                  className="btn btn-primary mt-2"
                  onClick={() => {
                    setShowForm(true);
                    setSelectedId(null);
                  }}
                >
                  Report your first incident
                </button>
              </div>
            ) : (
              <div className="card-list">
                {incidents.map((inc) => (
                  <IncidentCard
                    key={inc.id}
                    incident={inc}
                    selected={inc.id === selectedId}
                    onClick={() => {
                      setSelectedId(inc.id);
                      setShowForm(false);
                    }}
                  />
                ))}
              </div>
            )}
          </div>

          {showForm && (
            <aside className="workspace__side card detail-panel">
              <div className="detail-header__top">
                <h2 className="detail-header__title">Report an incident</h2>
                <button
                  type="button"
                  className="btn btn-sm btn-outline"
                  onClick={() => {
                    setShowForm(false);
                    resetForm();
                  }}
                  aria-label="Close the report form"
                >
                  ✕
                </button>
              </div>

              <p className="system-note mb-2">
                <span aria-hidden="true">⚙</span> You don&apos;t need to pick an
                urgency. The system assigns the priority and response deadline
                from the category and details you provide.
              </p>

              {formError && (
                <div className="alert alert-error mb-2" role="alert">
                  {formError}
                </div>
              )}

              <form onSubmit={handleSubmit} className="form-stack">
                <div className="form-group">
                  <label className="form-label" htmlFor="field-title">
                    Title <span className="req">*</span>
                  </label>
                  <input
                    id="field-title"
                    className="form-input"
                    placeholder="Brief description of the issue"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                    required
                    maxLength={200}
                  />
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="field-description">
                    Description
                  </label>
                  <textarea
                    id="field-description"
                    className="form-textarea"
                    placeholder="Where exactly is it, how bad is it, is anyone at risk?"
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    maxLength={2000}
                  />
                  <p className="form-hint">{description.length}/2000 characters</p>
                </div>

                <div className="form-group">
                  <label className="form-label" htmlFor="field-category">
                    Category
                  </label>
                  <select
                    id="field-category"
                    className="form-select"
                    value={category}
                    onChange={(e) => setCategory(e.target.value)}
                  >
                    {CATEGORIES.map((c) => (
                      <option key={c} value={c}>
                        {CATEGORY_LABELS[c] ?? c}
                      </option>
                    ))}
                  </select>
                  <p className="form-hint">
                    The category drives the assigned priority and SLA.
                  </p>
                </div>

                <div className="form-group">
                  <span className="form-label">
                    Location <span className="req">*</span>
                  </span>
                  <LocationCapture onLocation={setLocation} />
                </div>

                <div className="form-group">
                  <span className="form-label">Photos (optional)</span>
                  <ImageUploadField onChange={setImages} />
                </div>

                <button
                  id="btn-submit-incident"
                  type="submit"
                  className="btn btn-primary btn-full"
                  disabled={submitting || !location}
                >
                  {submitting ? (
                    <>
                      <span className="spinner spinner-sm" /> Submitting…
                    </>
                  ) : (
                    'Submit report'
                  )}
                </button>
              </form>
            </aside>
          )}

          {selected && !showForm && (
            <aside className="workspace__side card detail-panel">
              <DetailHeader incident={selected} onClose={() => setSelectedId(null)} />
              <WorkflowTimeline status={selected.status} />
              <PrioritySummary incident={selected} />
              <SlaSummary incident={selected}>
                <p className="system-note">
                  <span aria-hidden="true">⚙</span> The response deadline is set
                  by the city and cannot be changed by reporters.
                </p>
              </SlaSummary>
              <IncidentMetadata incident={selected} showReporter={false} />
              <IncidentImages
                incidentId={selected.id}
                expectedCount={selected.image_count}
              />
            </aside>
          )}
        </div>
      </main>
    </>
  );
}
