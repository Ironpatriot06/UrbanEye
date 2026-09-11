'use client';
/* ======================================================
   Authenticated incident image viewing.

   Incident images are protected by the same JWT as the rest of the API:
   USER sees their own incidents' images, AGENT sees images on incidents
   assigned to them, ADMIN sees everything. An <img src> cannot send an
   Authorization header, so each image is fetched with the bearer token and
   rendered from a blob object URL, which is revoked on unmount.
   ====================================================== */

import { useCallback, useEffect, useRef, useState } from 'react';
import { ImageMeta, apiGetImages, fetchImageObjectUrl } from '@/lib/api';

type LoadState = 'loading' | 'ready' | 'error';

/** One authenticated thumbnail. Reports its object URL up for the lightbox. */
function AuthImage({
  incidentId,
  image,
  onOpen,
}: {
  incidentId: number;
  image: ImageMeta;
  onOpen: (url: string, alt: string) => void;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [state, setState] = useState<LoadState>('loading');

  useEffect(() => {
    const controller = new AbortController();
    let objectUrl: string | null = null;

    fetchImageObjectUrl(incidentId, image.id, controller.signal)
      .then((created) => {
        objectUrl = created;
        setUrl(created);
        setState('ready');
      })
      .catch((err: unknown) => {
        if ((err as { name?: string })?.name === 'AbortError') return;
        setState('error');
      });

    return () => {
      controller.abort();
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [incidentId, image.id]);

  const alt = `Photo attached to incident #${incidentId}: ${image.filename}`;

  if (state === 'loading') {
    return (
      <div className="img-thumb img-thumb--placeholder" role="status" aria-label="Loading photo">
        <span className="spinner spinner-sm" />
      </div>
    );
  }

  if (state === 'error' || !url) {
    return (
      <div className="img-thumb img-thumb--error" role="img" aria-label={`${image.filename} failed to load`}>
        <span aria-hidden="true">⚠</span>
        <span className="img-thumb__caption">Unavailable</span>
      </div>
    );
  }

  return (
    <button
      type="button"
      className="img-thumb-button"
      onClick={() => onOpen(url, alt)}
      title={`${image.filename} — open larger view`}
    >
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt={alt} className="img-thumb" />
    </button>
  );
}

/** Full-viewport lightbox. Closes on backdrop click or Escape. */
function Lightbox({
  url,
  alt,
  onClose,
}: {
  url: string;
  alt: string;
  onClose: () => void;
}) {
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKey);
    closeRef.current?.focus();
    return () => document.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div
      className="lightbox"
      role="dialog"
      aria-modal="true"
      aria-label="Incident photo"
      onClick={onClose}
    >
      <button ref={closeRef} type="button" className="lightbox__close" onClick={onClose}>
        Close ✕
      </button>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={url} alt={alt} className="lightbox__img" onClick={(e) => e.stopPropagation()} />
    </div>
  );
}

/**
 * Gallery of every image attached to an incident.
 *
 * Renders nothing at all when the incident has no images and none are
 * expected, so callers can drop it in unconditionally.
 */
export function IncidentImages({
  incidentId,
  expectedCount,
}: {
  incidentId: number;
  expectedCount?: number;
}) {
  const [images, setImages] = useState<ImageMeta[]>([]);
  const [state, setState] = useState<LoadState>('loading');
  const [lightbox, setLightbox] = useState<{ url: string; alt: string } | null>(null);

  useEffect(() => {
    let cancelled = false;
    setState('loading');
    apiGetImages(incidentId)
      .then((imgs) => {
        if (cancelled) return;
        setImages(imgs);
        setState('ready');
      })
      .catch(() => {
        if (!cancelled) setState('error');
      });
    return () => {
      cancelled = true;
    };
  }, [incidentId]);

  const openLightbox = useCallback((url: string, alt: string) => {
    setLightbox({ url, alt });
  }, []);

  if (state === 'ready' && images.length === 0 && !expectedCount) return null;

  return (
    <section className="detail-section">
      <h4 className="detail-section__title">
        Photos{state === 'ready' && images.length > 0 ? ` (${images.length})` : ''}
      </h4>

      {state === 'loading' && (
        <div className="thumb-row" aria-busy="true">
          {Array.from({ length: Math.max(expectedCount ?? 1, 1) }).map((_, i) => (
            <div key={i} className="img-thumb img-thumb--placeholder">
              <span className="spinner spinner-sm" />
            </div>
          ))}
        </div>
      )}

      {state === 'error' && (
        <p className="alert alert-error alert-sm">
          Could not load the photos for this incident.
        </p>
      )}

      {state === 'ready' && images.length === 0 && (
        <p className="muted-note">No photos were attached to this report.</p>
      )}

      {state === 'ready' && images.length > 0 && (
        <div className="thumb-row">
          {images.map((img) => (
            <AuthImage
              key={img.id}
              incidentId={incidentId}
              image={img}
              onOpen={openLightbox}
            />
          ))}
        </div>
      )}

      {lightbox && (
        <Lightbox
          url={lightbox.url}
          alt={lightbox.alt}
          onClose={() => setLightbox(null)}
        />
      )}
    </section>
  );
}
