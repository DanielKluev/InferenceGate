import type { ReactNode } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { LayoutDashboard, Database } from 'lucide-react';

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
      <nav className="bg-card border-b shadow-md sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center">
                  <Database className="w-5 h-5 text-primary-foreground" />
                </div>
                <h1 className="text-xl font-bold text-foreground">InferenceGate</h1>
              </div>
              <div className="flex items-center gap-2">
                <Button
                  asChild
                  variant={isActive('/') ? 'default' : 'ghost'}
                  size="sm"
                >
                  <Link to="/" className="flex items-center gap-2">
                    <LayoutDashboard className="w-4 h-4" />
                    Dashboard
                  </Link>
                </Button>
                <Button
                  asChild
                  variant={isActive('/cache') || location.pathname.startsWith('/cache/') ? 'default' : 'ghost'}
                  size="sm"
                >
                  <Link to="/cache" className="flex items-center gap-2">
                    <Database className="w-4 h-4" />
                    Cache Entries
                  </Link>
                </Button>
              </div>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        {children}
      </main>
    </div>
  );
}
