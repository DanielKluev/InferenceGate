import { useEffect, useState } from 'react';
import { getCacheList } from '../api/client';
import type { CacheEntry } from '../api/client';
import CacheTable from '../components/CacheTable';
import { Skeleton } from '@/components/ui/skeleton';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';

export default function CacheList() {
  const [entries, setEntries] = useState<CacheEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchEntries() {
      try {
        const data = await getCacheList();
        setEntries(data);
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to fetch entries');
      } finally {
        setLoading(false);
      }
    }
    fetchEntries();
  }, []);

  if (loading) {
    return (
      <div className="px-4 sm:px-0">
        <h2 className="text-2xl font-bold mb-6">Cache Entries</h2>
        <Card>
          <CardHeader>
            <Skeleton className="h-10 w-full" />
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {[1, 2, 3, 4, 5].map((i) => (
                <Skeleton key={i} className="h-12 w-full" />
              ))}
            </div>
          </CardContent>
        </Card>
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

  return (
    <div className="px-4 sm:px-0">
      <h2 className="text-2xl font-bold mb-6">Cache Entries</h2>
      <CacheTable entries={entries} />
    </div>
  );
}
