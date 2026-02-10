import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCacheEntry } from '../api/client';
import type { CacheEntryDetail } from '../api/client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowLeft, AlertCircle, FileCode, Mail, Package } from 'lucide-react';

export default function EntryDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [entry, setEntry] = useState<CacheEntryDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchEntry() {
      if (!id) {
        setError('No entry ID provided');
        setLoading(false);
        return;
      }
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
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <Skeleton className="h-10 w-64" />
          <Skeleton className="h-10 w-32" />
        </div>
        <div className="space-y-6">
          <Card>
            <CardHeader>
              <Skeleton className="h-6 w-24" />
            </CardHeader>
            <CardContent>
              <Skeleton className="h-32 w-full" />
            </CardContent>
          </Card>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button
            onClick={() => navigate('/cache')}
            variant="ghost"
            size="sm"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <h2 className="text-3xl font-bold tracking-tight">Entry Detail</h2>
        </div>
        <Alert variant="destructive">
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Error</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="space-y-6">
        <div className="flex items-center gap-4">
          <Button
            onClick={() => navigate('/cache')}
            variant="ghost"
            size="sm"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <h2 className="text-3xl font-bold tracking-tight">Entry Detail</h2>
        </div>
        <Alert>
          <AlertCircle className="h-4 w-4" />
          <AlertTitle>Not Found</AlertTitle>
          <AlertDescription>The requested cache entry was not found.</AlertDescription>
        </Alert>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Button
            onClick={() => navigate('/cache')}
            variant="ghost"
            size="sm"
          >
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back
          </Button>
          <div>
            <h2 className="text-3xl font-bold tracking-tight">Entry Detail</h2>
            <p className="text-muted-foreground mt-1">Cache entry {entry.id.substring(0, 8)}...</p>
          </div>
        </div>
      </div>

      <div className="space-y-6">
        {/* Metadata */}
        <Card className="hover:shadow-lg transition-shadow duration-200">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileCode className="w-5 h-5" />
              Metadata
            </CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <div className="p-3 rounded-lg bg-muted/50">
                <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">ID</dt>
                <dd className="text-sm font-mono truncate">{entry.id}</dd>
              </div>
              {entry.model && (
                <div className="p-3 rounded-lg bg-muted/50">
                  <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Model</dt>
                  <dd className="text-sm font-semibold">{entry.model}</dd>
                </div>
              )}
              {entry.temperature !== null && (
                <div className="p-3 rounded-lg bg-muted/50">
                  <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Temperature</dt>
                  <dd className="text-sm font-semibold">{entry.temperature}</dd>
                </div>
              )}
              {entry.prompt_hash && (
                <div className="p-3 rounded-lg bg-muted/50">
                  <dt className="text-xs font-medium text-muted-foreground uppercase tracking-wider mb-1">Prompt Hash</dt>
                  <dd className="text-sm font-mono truncate">{entry.prompt_hash}</dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>

        <Separator />

        {/* Request and Response Tabs */}
        <Tabs defaultValue="request" className="w-full">
          <TabsList className="grid w-full grid-cols-3 h-12">
            <TabsTrigger value="request" className="flex items-center gap-2">
              <Mail className="w-4 h-4" />
              Request
            </TabsTrigger>
            <TabsTrigger value="response" className="flex items-center gap-2">
              <Package className="w-4 h-4" />
              Response
            </TabsTrigger>
            {entry.response.chunks && entry.response.chunks.length > 0 && (
              <TabsTrigger value="chunks" className="flex items-center gap-2">
                <FileCode className="w-4 h-4" />
                Chunks ({entry.response.chunks.length})
              </TabsTrigger>
            )}
          </TabsList>

          <TabsContent value="request" className="mt-6">
            <Card className="hover:shadow-lg transition-shadow duration-200">
              <CardHeader>
                <CardTitle>Request Details</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-6">
                  <div>
                    <dt className="text-sm font-medium text-muted-foreground mb-2">Method & Path</dt>
                    <dd className="flex items-center gap-2">
                      <Badge variant="secondary" className="font-mono">
                        {entry.request.method}
                      </Badge>
                      <span className="font-mono text-sm">{entry.request.path}</span>
                    </dd>
                  </div>
                  {entry.request.headers && Object.keys(entry.request.headers).length > 0 && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground mb-2">Headers</dt>
                      <dd>
                        <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto border">
                          {JSON.stringify(entry.request.headers, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                  {entry.request.body && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground mb-2">Body</dt>
                      <dd>
                        <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto border">
                          {JSON.stringify(entry.request.body, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                </dl>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="response" className="mt-6">
            <Card className="hover:shadow-lg transition-shadow duration-200">
              <CardHeader>
                <CardTitle>Response Details</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-6">
                  <div>
                    <dt className="text-sm font-medium text-muted-foreground mb-2">Status Code</dt>
                    <dd className="flex items-center gap-2">
                      <Badge
                        variant={entry.response.status_code === 200 ? 'default' : 'destructive'}
                        className="font-mono"
                      >
                        {entry.response.status_code}
                      </Badge>
                      {entry.response.is_streaming && (
                        <Badge variant="outline" className="border-primary/50 text-primary">
                          Streaming
                        </Badge>
                      )}
                    </dd>
                  </div>
                  {entry.response.headers && Object.keys(entry.response.headers).length > 0 && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground mb-2">Headers</dt>
                      <dd>
                        <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto border">
                          {JSON.stringify(entry.response.headers, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                  {entry.response.body && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground mb-2">Body</dt>
                      <dd>
                        <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto border">
                          {JSON.stringify(entry.response.body, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                </dl>
              </CardContent>
            </Card>
          </TabsContent>

          {entry.response.chunks && entry.response.chunks.length > 0 && (
            <TabsContent value="chunks" className="mt-6">
              <Card className="hover:shadow-lg transition-shadow duration-200">
                <CardHeader>
                  <CardTitle>Streaming Chunks ({entry.response.chunks.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs bg-muted p-4 rounded-lg overflow-x-auto border max-h-96">
                    {entry.response.chunks.join('')}
                  </pre>
                </CardContent>
              </Card>
            </TabsContent>
          )}
        </Tabs>
      </div>
    </div>
  );
}
