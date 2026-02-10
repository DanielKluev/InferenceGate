import { useEffect, useState } from 'react';
import { getCacheStats, getConfig } from '../api/client';
import type { CacheStats, Config } from '../api/client';
import StatsCards from '../components/StatsCards';

export default function Dashboard() {
  const [stats, setStats] = useState<CacheStats | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        const [statsData, configData] = await Promise.all([
          getCacheStats(),
          getConfig(),
        ]);
        setStats(statsData);
        setConfig(configData);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch data');
      } finally {
        setLoading(false);
      }
    }
    fetchData();
  }, []);

  if (loading) {
    return (
      <div className="flex justify-center items-center h-64">
        <div className="text-gray-500">Loading...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-md p-4">
        <p className="text-red-800">Error: {error}</p>
      </div>
    );
  }

  const formatBytes = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const statsCards = [
    {
      title: 'Total Entries',
      value: stats?.total_entries ?? 0,
    },
    {
      title: 'Cache Size',
      value: formatBytes(stats?.total_size_bytes ?? 0),
    },
    {
      title: 'Streaming Responses',
      value: stats?.streaming_responses ?? 0,
    },
    {
      title: 'Current Mode',
      value: config?.mode ?? 'unknown',
    },
  ];

  return (
    <div className="px-4 sm:px-0">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Dashboard</h2>
      
      <StatsCards stats={statsCards} />

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Configuration</h3>
          <dl className="space-y-2">
            <div className="flex justify-between">
              <dt className="text-sm text-gray-500">Mode:</dt>
              <dd className="text-sm font-medium text-gray-900">{config?.mode}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm text-gray-500">Proxy:</dt>
              <dd className="text-sm font-medium text-gray-900">
                {config?.host}:{config?.port}
              </dd>
            </div>
            {config?.upstream_url && (
              <div className="flex justify-between">
                <dt className="text-sm text-gray-500">Upstream:</dt>
                <dd className="text-sm font-medium text-gray-900">{config?.upstream_url}</dd>
              </div>
            )}
            <div className="flex justify-between">
              <dt className="text-sm text-gray-500">Cache Directory:</dt>
              <dd className="text-sm font-medium text-gray-900 font-mono">{config?.cache_dir}</dd>
            </div>
          </dl>
        </div>

        <div className="bg-white shadow rounded-lg p-6">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Entries by Model</h3>
          {stats && Object.keys(stats.entries_by_model).length > 0 ? (
            <dl className="space-y-2">
              {Object.entries(stats.entries_by_model).map(([model, count]) => (
                <div key={model} className="flex justify-between">
                  <dt className="text-sm text-gray-500">{model}:</dt>
                  <dd className="text-sm font-medium text-gray-900">{count}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <p className="text-sm text-gray-500">No entries yet</p>
          )}
        </div>
      </div>
    </div>
  );
}
