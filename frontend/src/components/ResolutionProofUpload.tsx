'use client';
/* ======================================================
   Proof of completed work — the agent's upload control.

   Shown on the agent dashboard once an incident they are assigned to has been
   marked Resolved. The photo goes to the same store as a citizen's report
   photo and becomes visible to the reporting citizen, the assigned agent and
   any admin.

   This component decides what is OFFERED, never what is permitted: the backend
   accepts the upload only from the assigned agent (or an admin) and only while
   the incident is RESOLVED, answering 403 or 422 otherwise.
   ====================================================== */

import { useRef, useState } from 'react';
import { apiUploadResolutionImage } from '@/lib/api';

const ACCEPTED = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_MB = 5;

export function ResolutionProofUpload({
  incidentId,
  onUploaded,
}: {
  incidentId: number;
  /** Called after each successful upload so the gallery re-fetches. */
  onUploaded: () => void;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [dragging, setDragging] = useState(false);

  const validate = (file: File): string | undefined => {
    if (!ACCEPTED.includes(file.type)) {
      return `${file.name}: unsupported type (JPEG, PNG or WEBP only)`;
    }
    if (file.size > MAX_MB * 1024 * 1024) {
      return `${file.name}: exceeds the ${MAX_MB} MB limit`;
    }
    return undefined;
  };

  const upload = async (files: File[]) => {
    if (files.length === 0) return;
    setError('');
    setNotice('');

    // Checked here so an obviously wrong file is refused without a round trip.
    // The server validates the same things again, including the magic bytes.
    const rejected = files.map(validate).filter(Boolean) as string[];
    if (rejected.length > 0) {
      setError(rejected.join('; '));
      return;
    }

    setBusy(true);
    try {
      for (const file of files) {
        await apiUploadResolutionImage(incidentId, file);
      }
      setNotice(
        files.length === 1
          ? 'Proof of work attached.'
          : `${files.length} photos attached.`,
      );
      onUploaded();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Could not attach the photo.');
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="detail-section">
      <h4 className="detail-section__title">Attach proof of work</h4>
      <p className="muted-note">
        Show what the finished job looks like. The person who reported this and
        your admin will both see it.
      </p>

      {error && (
        <div className="alert alert-error alert-sm mb-2" role="alert">
          {error}
        </div>
      )}
      {notice && (
        <div className="alert alert-success alert-sm mb-2" role="status">
          {notice}
        </div>
      )}

      <button
        type="button"
        id="btn-upload-proof"
        className={`dropzone dropzone--compact${dragging ? ' is-dragging' : ''}`}
        onClick={() => inputRef.current?.click()}
        disabled={busy}
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!busy) upload(Array.from(e.dataTransfer.files));
        }}
      >
        {busy ? (
          <>
            <span className="spinner spinner-sm" />
            <span className="dropzone__title">Uploading…</span>
          </>
        ) : (
          <>
            <span className="dropzone__icon" aria-hidden="true">🛠</span>
            <span className="dropzone__title">Click or drag to add proof photos</span>
            <span className="dropzone__hint">
              JPEG, PNG or WEBP · max {MAX_MB} MB each
            </span>
          </>
        )}
      </button>

      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED.join(',')}
        multiple
        hidden
        id="proof-upload-input"
        onChange={(e) => {
          upload(Array.from(e.target.files || []));
          e.target.value = '';
        }}
      />
    </section>
  );
}
