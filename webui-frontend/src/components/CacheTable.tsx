import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import type { CacheEntry } from '../api/client';
import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';

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
    <Card>
      <CardHeader>
        <Input
          type="text"
          placeholder="Search entries..."
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
        />
      </CardHeader>
      <CardContent className="p-0">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort('id')}
              >
                ID {sortField === 'id' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort('model')}
              >
                Model {sortField === 'model' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort('method')}
              >
                Method {sortField === 'method' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort('path')}
              >
                Path {sortField === 'path' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead
                className="cursor-pointer hover:bg-muted/50"
                onClick={() => handleSort('status_code')}
              >
                Status {sortField === 'status_code' && (sortDirection === 'asc' ? '↑' : '↓')}
              </TableHead>
              <TableHead>
                Type
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sortedEntries.map((entry) => (
              <TableRow
                key={entry.id}
                className="cursor-pointer"
                onClick={() => handleRowClick(entry.id)}
              >
                <TableCell className="font-mono text-muted-foreground">
                  {entry.id.substring(0, 8)}...
                </TableCell>
                <TableCell>
                  {entry.model || '-'}
                </TableCell>
                <TableCell>
                  <Badge variant="secondary">
                    {entry.method}
                  </Badge>
                </TableCell>
                <TableCell className="text-muted-foreground">
                  {entry.path}
                </TableCell>
                <TableCell>
                  <Badge
                    variant={entry.status_code === 200 ? 'default' : 'destructive'}
                  >
                    {entry.status_code}
                  </Badge>
                </TableCell>
                <TableCell>
                  {entry.is_streaming ? (
                    <Badge variant="outline">
                      Streaming
                    </Badge>
                  ) : (
                    <Badge variant="secondary">
                      Standard
                    </Badge>
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {sortedEntries.length === 0 && (
          <div className="text-center py-12">
            <p className="text-muted-foreground">No entries found.</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
