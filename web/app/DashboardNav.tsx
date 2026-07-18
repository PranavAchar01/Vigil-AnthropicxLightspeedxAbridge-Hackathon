import Link from "next/link";

type DashboardRoute = "overview" | "clinical" | "operations" | "trust";

const routes: Array<{ id: DashboardRoute; href: string; label: string }> = [
  { id: "overview", href: "/", label: "Overview" },
  { id: "clinical", href: "/clinical", label: "Clinical" },
  { id: "operations", href: "/operations", label: "Operations" },
  { id: "trust", href: "/trust", label: "Trust & audit" },
];

export default function DashboardNav({ current }: { current: DashboardRoute }) {
  return (
    <nav className="product-nav surface" aria-label="Vigil dashboards">
      <div className="product-nav-lockup">
        <Link className="product-nav-brand" href="/" aria-label="Vigil overview">
          <span aria-hidden="true"><i /></span>
          Vigil
        </Link>
        <small>Waiting room 01</small>
      </div>
      <div className="product-nav-routes">
        {routes.map((route) => <Link className={current === route.id ? "active" : ""} href={route.href} aria-current={current === route.id ? "page" : undefined} key={route.id}>{route.label}</Link>)}
      </div>
      <span className="product-nav-context"><i /> Privacy mode</span>
    </nav>
  );
}
