import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCacheEntry } from '../api/client';
import type { CacheEntryDetail } from '../api/client';

export default function EntryDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [entry, setEntry] = useState<CacheEntryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchEntry() {
      if (!id) return;
      try {
        const data = await getCacheEntry(id);
        setEntry(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch entry');
      } finally {
        setLoading(false);
      }
    }
    fetchEntry();
  }, [id]);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-4 sm:px-0">
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <p className="text-red-800">Error: {error}</p>
        </div>
        <button
          onClick={() => navigate('/cache')}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Back to Cache List
        </button>
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="px-4 sm:px-0">
        <div className="bg-yellow-50 border border-yellow-200 rounded-md p-4">
          <p className="text-yellow-800">Entry not found</p>
        </div>
        <button
          onClick={() => navigate('/cache')}
          className="mt-4 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700"
        >
          Back to Cache List
        </button>
      </div>
    );
  }

  return (
    <div className="px-4 sm:px-0">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold text-gray-900">Cache Entry Detail</h2>
        <button
          onClick={() => navigate('/cache')}
          className="px-4 py-2 bg-gray-600 text-white rounded hover:bg-gray-700"
        >
          Back to List
        </button>
      </div>

      <div className="space-y-6">
        {/* Metadata */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Metadata</h3>
          <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
            <div>
              <dt className="text-sm font-medium text-gray-500">ID</dt>
              <dd className="mt-1 text-sm text-gray-900 font-mono">{entry.id}</dd>
            </div>
            {entry.model && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Model</dt>
                <dd className="mt-1 text-sm text-gray-900">{entry.model}</dd>
              </div>
            )}
            {entry.temperature !== null && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Temperature</dt>
                <dd className="mt-1 text-sm text-gray-900">{entry.temperature}</dd>
              </div>
            )}
            {entry.prompt_hash && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Prompt Hash</dt>
                <dd className="mt-1 text-sm text-gray-900 font-mono">{entry.prompt_hash}</dd>
              </div>
            )}
          </dl>
        </div>

        {/* Request */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Request</h3>
          <dl className="space-y-4">
            <div>
              <dt className="text-sm font-medium text-gray-500">Method & Path</dt>
              <dd className="mt-1 text-sm text-gray-900">
                <span className="px-2 py-1 bg-blue-100 text-blue-800 rounded font-medium">
                  {entry.request.method}
                </span>
                <span className="ml-2">{entry.request.path}</span>
              </dd>
            </div>
            {entry.request.headers && Object.keys(entry.request.headers).length > 0 && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Headers</dt>
                <dd className="mt-1">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(entry.request.headers, null, 2)}
                  </pre>
                </dd>
              </div>
            )}
            {entry.request.body && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Body</dt>
                <dd className="mt-1">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(entry.request.body, null, 2)}
                  </pre>
                </dd>
              </div>
            )}
          </dl>
        </div>

        {/* Response */}
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Response</h3>
          <dl className="space-y-4">
            <div>
              <dt className="text-sm font-medium text-gray-500">Status Code</dt>
              <dd className="mt-1 text-sm text-gray-900">
                <span
                  className={`px-2 py-1 rounded font-medium ${
                    entry.response.status_code === 200
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {entry.response.status_code}
                </span>
                {entry.response.is_streaming && (
                  <span className="ml-2 px-2 py-1 bg-purple-100 text-purple-800 rounded font-medium">
                    Streaming
                  </span>
                )}
              </dd>
            </div>
            {entry.response.headers && Object.keys(entry.response.headers).length > 0 && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Headers</dt>
                <dd className="mt-1">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(entry.response.headers, null, 2)}
                  </pre>
                </dd>
              </div>
            )}
            {entry.response.body && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Body</dt>
                <dd className="mt-1">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {JSON.stringify(entry.response.body, null, 2)}
                  </pre>
                </dd>
              </div>
            )}
            {entry.response.chunks && entry.response.chunks.length > 0 && (
              <div>
                <dt className="text-sm font-medium text-gray-500">Chunks ({entry.response.chunks.length})</dt>
                <dd className="mt-1">
                  <pre className="text-xs bg-gray-50 p-3 rounded overflow-x-auto">
                    {entry.response.chunks.join('')}
                  </pre>
                </dd>
              </div>
            )}
          </dl>
        </div>
      </div>
    </div>
  );
}
