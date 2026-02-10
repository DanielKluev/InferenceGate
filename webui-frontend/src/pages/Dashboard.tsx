import { useEffect, useState } from 'react';
import { getCacheStats, getConfig } from '../api/client';
import type { CacheStats, Config } from '../api/client';
import StatsCards from '../components/StatsCards';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';

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
      <div className="px-4 sm:px-0">
        <h2 className="text-2xl font-bold mb-6">Dashboard</h2>
        <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader className="pb-2">
                <Skeleton className="h-4 w-24" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-16" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="px-4 sm:px-0">
        <Alert variant="destructive">
          <AlertDescription>Error: {error}</AlertDescription>
        </Alert>
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
      <h2 className="text-2xl font-bold mb-6">Dashboard</h2>
      
      <StatsCards stats={statsCards} />

      <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-2">
              <div className="flex justify-between">
                <dt className="text-sm text-muted-foreground">Mode:</dt>
                <dd className="text-sm font-medium">{config?.mode}</dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-muted-foreground">Proxy:</dt>
                <dd className="text-sm font-medium">
                  {config?.host}:{config?.port}
                </dd>
              </div>
              {config?.upstream_url && (
                <div className="flex justify-between">
                  <dt className="text-sm text-muted-foreground">Upstream:</dt>
                  <dd className="text-sm font-medium">{config?.upstream_url}</dd>
                </div>
              )}
              <div className="flex justify-between">
                <dt className="text-sm text-muted-foreground">Cache Directory:</dt>
                <dd className="text-sm font-medium font-mono">{config?.cache_dir}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Entries by Model</CardTitle>
          </CardHeader>
          <CardContent>
            {stats && Object.keys(stats.entries_by_model).length > 0 ? (
              <dl className="space-y-2">
                {Object.entries(stats.entries_by_model).map(([model, count]) => (
                  <div key={model} className="flex justify-between">
                    <dt className="text-sm text-muted-foreground">{model}:</dt>
                    <dd className="text-sm font-medium">{count}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground">No entries yet</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
