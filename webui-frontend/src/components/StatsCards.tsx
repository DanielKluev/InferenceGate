import { Card, CardContent, CardHeader } from '@/components/ui/card';

interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
}

export default function StatsCards({ stats }: { stats: StatCardProps[] }) {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, index) => (
        <Card key={index}>
          <CardHeader className="pb-2">
            <div className="text-sm font-medium text-muted-foreground">{stat.title}</div>
          </CardHeader>
          <CardContent>
            <div className="text-3xl font-semibold">{stat.value}</div>
            {stat.subtitle && (
              <p className="text-sm text-muted-foreground mt-1">{stat.subtitle}</p>
            )}
          </CardContent>
        </Card>
      ))}
    </div>
  );
}
