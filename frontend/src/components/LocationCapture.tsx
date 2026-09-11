'use client';
import { useState, useCallback } from 'react';

interface LocationResult {
  latitude: number;
  longitude: number;
}

interface Props {
  onLocation: (loc: LocationResult) => void;
}

export function LocationCapture({ onLocation }: Props) {
  const [status, setStatus] = useState<
    'idle' | 'loading' | 'success' | 'denied' | 'unavailable'
  >('idle');
  const [location, setLocation] = useState<LocationResult | null>(null);
  const [manual, setManual] = useState({ lat: '', lon: '' });
  const [showManual, setShowManual] = useState(false);
  const [manualError, setManualError] = useState('');

  const capture = useCallback(() => {
    if (!navigator.geolocation) {
      setStatus('unavailable');
      setShowManual(true);
      return;
    }
    setStatus('loading');
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const loc = {
          latitude: pos.coords.latitude,
          longitude: pos.coords.longitude,
        };
        setLocation(loc);
        setStatus('success');
        onLocation(loc);
      },
      () => {
        setStatus('denied');
        setShowManual(true);
      },
      { timeout: 10000, enableHighAccuracy: true },
    );
  }, [onLocation]);

  const handleManual = () => {
    const lat = parseFloat(manual.lat);
    const lon = parseFloat(manual.lon);
    if (Number.isNaN(lat) || lat < -90 || lat > 90) {
      setManualError('Latitude must be a number between −90 and 90.');
      return;
    }
    if (Number.isNaN(lon) || lon < -180 || lon > 180) {
      setManualError('Longitude must be a number between −180 and 180.');
      return;
    }
    setManualError('');
    const loc = { latitude: lat, longitude: lon };
    setLocation(loc);
    setStatus('success');
    onLocation(loc);
  };

  return (
    <div className="form-stack" style={{ gap: '.75rem' }}>
      {status !== 'success' && (
        <button
          type="button"
          className="btn btn-outline"
          onClick={capture}
          disabled={status === 'loading'}
          id="btn-capture-location"
        >
          {status === 'loading' ? (
            <>
              <span className="spinner spinner-sm" /> Detecting…
            </>
          ) : (
            <>
              <span aria-hidden="true">📍</span> Use my current location
            </>
          )}
        </button>
      )}

      {status === 'success' && location && (
        <div className="alert alert-success location-result" role="status">
          <span>
            ✓ Location set —{' '}
            <code>
              {location.latitude.toFixed(6)}, {location.longitude.toFixed(6)}
            </code>
          </span>
          <button
            type="button"
            className="btn btn-sm btn-outline"
            onClick={() => {
              setStatus('idle');
              setLocation(null);
              setShowManual(false);
            }}
          >
            Change
          </button>
        </div>
      )}

      {(status === 'denied' || status === 'unavailable') && (
        <div className="alert alert-warning alert-sm" role="alert">
          {status === 'denied'
            ? 'Location permission was denied. Enter the coordinates manually below.'
            : 'Your browser does not support geolocation. Enter the coordinates manually below.'}
        </div>
      )}

      {showManual && status !== 'success' && (
        <>
          <div className="location-manual">
            <div className="form-group">
              <label className="form-label" htmlFor="manual-lat">
                Latitude (−90 to 90)
              </label>
              <input
                id="manual-lat"
                className="form-input"
                type="number"
                step="any"
                placeholder="12.9716"
                value={manual.lat}
                onChange={(e) => setManual((m) => ({ ...m, lat: e.target.value }))}
              />
            </div>
            <div className="form-group">
              <label className="form-label" htmlFor="manual-lon">
                Longitude (−180 to 180)
              </label>
              <input
                id="manual-lon"
                className="form-input"
                type="number"
                step="any"
                placeholder="77.5946"
                value={manual.lon}
                onChange={(e) => setManual((m) => ({ ...m, lon: e.target.value }))}
              />
            </div>
            <button type="button" className="btn btn-primary" onClick={handleManual}>
              Set
            </button>
          </div>
          {manualError && (
            <div className="alert alert-error alert-sm" role="alert">
              {manualError}
            </div>
          )}
        </>
      )}

      {!showManual && status === 'idle' && (
        <button
          type="button"
          className="btn btn-sm btn-outline"
          style={{ alignSelf: 'flex-start' }}
          onClick={() => setShowManual(true)}
        >
          Enter coordinates manually
        </button>
      )}
    </div>
  );
}
