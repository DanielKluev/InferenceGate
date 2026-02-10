import { useEffect, useState } from 'react';
import { getCacheList } from '../api/client';
import type { CacheEntry } from '../api/client';
import CacheTable from '../components/CacheTable';

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

  return (
    <div className="px-4 sm:px-0">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Cache Entries</h2>
      <CacheTable entries={entries} />
    </div>
  );
}
