import { useEffect, useState } from 'react';
import { getCacheStats, getConfig } from '../api/client';
import type { CacheStats, Config } from '../api/client';
import StatsCards from '../components/StatsCards';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { AlertCircle } from 'lucide-react';

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
      <div className="space-y-8">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">Dashboard</h2>
          <p className="text-muted-foreground">Monitor your inference cache performance</p>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {[1, 2, 3, 4].map((i) => (
            <Card key={i}>
              <CardHeader className="pb-3">
                <Skeleton className="h-4 w-32" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-10 w-24" />
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div>
          <h2 className="text-3xl font-bold tracking-tight mb-2">Dashboard</h2>
          <p className="text-muted-foreground">Monitor your inference cache performance</p>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
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
    <div className="space-y-8">
      <div>
        <h2 className="text-3xl font-bold tracking-tight mb-2">Dashboard</h2>
        <p className="text-muted-foreground">Monitor your inference cache performance</p>
      </div>
      
      <StatsCards stats={statsCards} />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        <Card className="hover:shadow-lg transition-shadow duration-200">
          <CardHeader>
            <CardTitle className="text-lg">Configuration</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="space-y-3">
              <div className="flex items-center justify-between py-2 border-b border-border/50">
                <dt className="text-sm font-medium text-muted-foreground">Mode</dt>
                <dd className="text-sm font-semibold">{config?.mode}</dd>
              </div>
              <div className="flex items-center justify-between py-2 border-b border-border/50">
                <dt className="text-sm font-medium text-muted-foreground">Proxy</dt>
                <dd className="text-sm font-mono">
                  {config?.host}:{config?.port}
                </dd>
              </div>
              {config?.upstream_url && (
                <div className="flex items-center justify-between py-2 border-b border-border/50">
                  <dt className="text-sm font-medium text-muted-foreground">Upstream</dt>
                  <dd className="text-sm font-mono truncate max-w-[200px]">{config?.upstream_url}</dd>
                </div>
              )}
              <div className="flex items-center justify-between py-2">
                <dt className="text-sm font-medium text-muted-foreground">Cache Directory</dt>
                <dd className="text-sm font-mono truncate max-w-[200px]">{config?.cache_dir}</dd>
              </div>
            </dl>
          </CardContent>
        </Card>

        <Card className="hover:shadow-lg transition-shadow duration-200">
          <CardHeader>
            <CardTitle className="text-lg">Entries by Model</CardTitle>
          </CardHeader>
          <CardContent>
            {stats && Object.keys(stats.entries_by_model).length > 0 ? (
              <dl className="space-y-3">
                {Object.entries(stats.entries_by_model).map(([model, count]) => (
                  <div key={model} className="flex items-center justify-between py-2 border-b border-border/50 last:border-0">
                    <dt className="text-sm font-medium text-muted-foreground truncate">{model}</dt>
                    <dd className="text-sm font-semibold ml-4">{count}</dd>
                  </div>
                ))}
              </dl>
            ) : (
              <p className="text-sm text-muted-foreground text-center py-8">No entries yet</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
