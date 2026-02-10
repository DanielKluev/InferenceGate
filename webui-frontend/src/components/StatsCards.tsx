interface StatCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
}

export default function StatsCards({ stats }: { stats: StatCardProps[] }) {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 lg:grid-cols-4">
      {stats.map((stat, index) => (
        <div key={index} className="bg-white overflow-hidden shadow rounded-lg">
          <div className="px-4 py-5 sm:p-6">
            <dt className="text-sm font-medium text-gray-500 truncate">{stat.title}</dt>
            <dd className="mt-1 text-3xl font-semibold text-gray-900">{stat.value}</dd>
            {stat.subtitle && (
              <dd className="mt-1 text-sm text-gray-500">{stat.subtitle}</dd>
            )}
          </div>
        </div>
      ))}
    </div>
  );
}
