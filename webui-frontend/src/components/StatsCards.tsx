import { Card, CardContent, CardHeader } from '@/components/ui/card';
import { TrendingUp, HardDrive, Radio, Settings, type LucideIcon } from 'lucide-react';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
}

const iconMap: Record<string, LucideIcon> = {
  'Total Entries': TrendingUp,
  'Cache Size': HardDrive,
  'Streaming Responses': Radio,
  'Current Mode': Settings,
};

export default function StatsCards({ stats }: { stats: StatCardProps[] }) {
  return (
    <div className="grid grid-cols-1 gap-6 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat) => {
        const Icon = iconMap[stat.title] || TrendingUp;
        return (
          <Card key={stat.title} className="hover:shadow-lg transition-shadow duration-200">
            <CardHeader className="pb-3">
              <div className="flex items-center justify-between">
                <div className="text-sm font-medium text-muted-foreground">{stat.title}</div>
                <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                  <Icon className="w-5 h-5 text-primary" />
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="text-3xl font-bold tracking-tight">{stat.value}</div>
              {stat.subtitle && (
                <p className="text-sm text-muted-foreground mt-2">{stat.subtitle}</p>
              )}
            </CardContent>
          </Card>
        );
      })}
    </div>
  );
}
