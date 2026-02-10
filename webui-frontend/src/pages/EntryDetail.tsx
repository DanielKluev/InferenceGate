import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { getCacheEntry } from '../api/client';
import type { CacheEntryDetail } from '../api/client';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { ArrowLeft } from 'lucide-react';

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
      <div className="px-4 sm:px-0">
        <div className="mb-6 flex items-center justify-between">
          <Skeleton className="h-8 w-48" />
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
      <div className="px-4 sm:px-0">
        <Alert variant="destructive" className="mb-4">
          <AlertDescription>Error: {error}</AlertDescription>
        </Alert>
        <Button
          onClick={() => navigate('/cache')}
          variant="ghost"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Cache List
        </Button>
      </div>
    );
  }

  if (!entry) {
    return (
      <div className="px-4 sm:px-0">
        <Alert className="mb-4">
          <AlertDescription>Entry not found</AlertDescription>
        </Alert>
        <Button
          onClick={() => navigate('/cache')}
          variant="ghost"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Cache List
        </Button>
      </div>
    );
  }

  return (
    <div className="px-4 sm:px-0">
      <div className="mb-6 flex items-center justify-between">
        <h2 className="text-2xl font-bold">Cache Entry Detail</h2>
        <Button
          onClick={() => navigate('/cache')}
          variant="ghost"
        >
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to List
        </Button>
      </div>

      <div className="space-y-6">
        {/* Metadata */}
        <Card>
          <CardHeader>
            <CardTitle>Metadata</CardTitle>
          </CardHeader>
          <CardContent>
            <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
              <div>
                <dt className="text-sm font-medium text-muted-foreground">ID</dt>
                <dd className="mt-1 text-sm font-mono">{entry.id}</dd>
              </div>
              {entry.model && (
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Model</dt>
                  <dd className="mt-1 text-sm">{entry.model}</dd>
                </div>
              )}
              {entry.temperature !== null && (
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Temperature</dt>
                  <dd className="mt-1 text-sm">{entry.temperature}</dd>
                </div>
              )}
              {entry.prompt_hash && (
                <div>
                  <dt className="text-sm font-medium text-muted-foreground">Prompt Hash</dt>
                  <dd className="mt-1 text-sm font-mono">{entry.prompt_hash}</dd>
                </div>
              )}
            </dl>
          </CardContent>
        </Card>

        <Separator />

        {/* Request and Response Tabs */}
        <Tabs defaultValue="request" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="request">Request</TabsTrigger>
            <TabsTrigger value="response">Response</TabsTrigger>
            {entry.response.chunks && entry.response.chunks.length > 0 && (
              <TabsTrigger value="chunks">Chunks ({entry.response.chunks.length})</TabsTrigger>
            )}
          </TabsList>

          <TabsContent value="request">
            <Card>
              <CardHeader>
                <CardTitle>Request</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-4">
                  <div>
                    <dt className="text-sm font-medium text-muted-foreground">Method & Path</dt>
                    <dd className="mt-1 text-sm">
                      <Badge variant="secondary">
                        {entry.request.method}
                      </Badge>
                      <span className="ml-2">{entry.request.path}</span>
                    </dd>
                  </div>
                  {entry.request.headers && Object.keys(entry.request.headers).length > 0 && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Headers</dt>
                      <dd className="mt-1">
                        <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                          {JSON.stringify(entry.request.headers, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                  {entry.request.body && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Body</dt>
                      <dd className="mt-1">
                        <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                          {JSON.stringify(entry.request.body, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                </dl>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="response">
            <Card>
              <CardHeader>
                <CardTitle>Response</CardTitle>
              </CardHeader>
              <CardContent>
                <dl className="space-y-4">
                  <div>
                    <dt className="text-sm font-medium text-muted-foreground">Status Code</dt>
                    <dd className="mt-1 text-sm">
                      <Badge
                        variant={entry.response.status_code === 200 ? 'default' : 'destructive'}
                      >
                        {entry.response.status_code}
                      </Badge>
                      {entry.response.is_streaming && (
                        <Badge variant="outline" className="ml-2">
                          Streaming
                        </Badge>
                      )}
                    </dd>
                  </div>
                  {entry.response.headers && Object.keys(entry.response.headers).length > 0 && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Headers</dt>
                      <dd className="mt-1">
                        <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
                          {JSON.stringify(entry.response.headers, null, 2)}
                        </pre>
                      </dd>
                    </div>
                  )}
                  {entry.response.body && (
                    <div>
                      <dt className="text-sm font-medium text-muted-foreground">Body</dt>
                      <dd className="mt-1">
                        <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
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
            <TabsContent value="chunks">
              <Card>
                <CardHeader>
                  <CardTitle>Streaming Chunks ({entry.response.chunks.length})</CardTitle>
                </CardHeader>
                <CardContent>
                  <pre className="text-xs bg-muted p-3 rounded overflow-x-auto">
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
