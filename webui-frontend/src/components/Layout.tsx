import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { Separator } from '@/components/ui/separator';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation();

  const isActive = (path: string) => {
    return location.pathname === path;
  };

  return (
    <div className="min-h-screen bg-background">
      <nav className="bg-card shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center">
                <h1 className="text-xl font-bold text-foreground">InferenceGate Dashboard</h1>
              </div>
              <div className="ml-6 flex space-x-2">
                <Button
                  asChild
                  variant={isActive('/') ? 'default' : 'ghost'}
                >
                  <Link to="/">Dashboard</Link>
                </Button>
                <Button
                  asChild
                  variant={isActive('/cache') || location.pathname.startsWith('/cache/') ? 'default' : 'ghost'}
                >
                  <Link to="/cache">Cache Entries</Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </nav>
      <Separator />
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}
