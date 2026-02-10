import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import type { CacheEntry } from '../api/client';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Search } from 'lucide-react';

interface CacheTableProps {
  entries: CacheEntry[];
}

export default function CacheTable({ entries }: CacheTableProps) {
  const navigate = useNavigate();
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<keyof CacheEntry>('id');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const filteredEntries = useMemo(() => {
    return entries.filter(entry => {
      const searchLower = searchTerm.toLowerCase();
      return (
        entry.id.toLowerCase().includes(searchLower) ||
        entry.model?.toLowerCase().includes(searchLower) ||
        entry.path.toLowerCase().includes(searchLower) ||
        entry.method.toLowerCase().includes(searchLower)
      );
    });
  }, [entries, searchTerm]);

  const sortedEntries = useMemo(() => {
    return [...filteredEntries].sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      if (aVal === null && bVal === null) return 0;
      if (aVal === null) return 1;
      if (bVal === null) return -1;
      
      const comparison = aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      return sortDirection === 'asc' ? comparison : -comparison;
    });
  }, [filteredEntries, sortField, sortDirection]);

  const handleSort = (field: keyof CacheEntry) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('asc');
    }
  };

  const handleRowClick = (id: string) => {
    navigate(`/cache/${id}`);
  };

  return (
    <Card className="hover:shadow-lg transition-shadow duration-200">
      <CardHeader className="pb-4">
        <div className="relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-muted-foreground w-4 h-4" aria-hidden="true" />
          <Input
            type="text"
            placeholder="Search by ID, model, path, or method..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="pl-10"
            aria-label="Search cache entries"
          />
        </div>
      </CardHeader>
      <CardContent className="p-0">
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead
                  className="font-semibold"
                  aria-sort={sortField === 'id' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button
                    type="button"
                    className="flex items-center gap-1 w-full text-left hover:bg-muted transition-colors px-2 py-1 rounded"
                    onClick={() => handleSort('id')}
                  >
                    ID {sortField === 'id' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </button>
                </TableHead>
                <TableHead
                  className="font-semibold"
                  aria-sort={sortField === 'model' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button
                    type="button"
                    className="flex items-center gap-1 w-full text-left hover:bg-muted transition-colors px-2 py-1 rounded"
                    onClick={() => handleSort('model')}
                  >
                    Model {sortField === 'model' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </button>
                </TableHead>
                <TableHead
                  className="font-semibold"
                  aria-sort={sortField === 'method' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button
                    type="button"
                    className="flex items-center gap-1 w-full text-left hover:bg-muted transition-colors px-2 py-1 rounded"
                    onClick={() => handleSort('method')}
                  >
                    Method {sortField === 'method' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </button>
                </TableHead>
                <TableHead
                  className="font-semibold"
                  aria-sort={sortField === 'path' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button
                    type="button"
                    className="flex items-center gap-1 w-full text-left hover:bg-muted transition-colors px-2 py-1 rounded"
                    onClick={() => handleSort('path')}
                  >
                    Path {sortField === 'path' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </button>
                </TableHead>
                <TableHead
                  className="font-semibold"
                  aria-sort={sortField === 'status_code' ? (sortDirection === 'asc' ? 'ascending' : 'descending') : 'none'}
                >
                  <button
                    type="button"
                    className="flex items-center gap-1 w-full text-left hover:bg-muted transition-colors px-2 py-1 rounded"
                    onClick={() => handleSort('status_code')}
                  >
                    Status {sortField === 'status_code' && (sortDirection === 'asc' ? '↑' : '↓')}
                  </button>
                </TableHead>
                <TableHead className="font-semibold">
                  Type
                </TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sortedEntries.map((entry) => (
                <TableRow
                  key={entry.id}
                  className="cursor-pointer hover:bg-muted/50 transition-colors"
                  onClick={() => handleRowClick(entry.id)}
                  tabIndex={0}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault();
                      handleRowClick(entry.id);
                    }
                  }}
                  role="button"
                >
                  <TableCell className="font-mono text-xs text-muted-foreground">
                    {entry.id.substring(0, 8)}...
                  </TableCell>
                  <TableCell className="font-medium">
                    {entry.model || '-'}
                  </TableCell>
                  <TableCell>
                    <Badge variant="secondary" className="font-mono text-xs">
                      {entry.method}
                    </Badge>
                  </TableCell>
                  <TableCell className="text-muted-foreground font-mono text-sm max-w-xs truncate">
                    {entry.path}
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant={entry.status_code === 200 ? 'default' : 'destructive'}
                      className="font-mono"
                    >
                      {entry.status_code}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    {entry.is_streaming ? (
                      <Badge variant="outline" className="border-primary/50 text-primary">
                        Streaming
                      </Badge>
                    ) : (
                      <Badge variant="secondary" className="text-muted-foreground">
                        Standard
                      </Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
        {sortedEntries.length === 0 && (
          <div className="text-center py-12 border-t">
            <p className="text-muted-foreground">No entries found.</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
