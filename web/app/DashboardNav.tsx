import Link from "next/link";

export type DashboardRoute = "command" | "front-desk" | "nurse" | "security";

const routes: Array<{ id: DashboardRoute; href: string; label: string }> = [
  { id: "command", href: "/", label: "Command center" },
  { id: "front-desk", href: "/front-desk", label: "Front desk" },
  { id: "nurse", href: "/nurse", label: "Nurse" },
  { id: "security", href: "/security", label: "Security" },
];

export default function DashboardNav({ current }: { current: DashboardRoute }) {
  return (
    <nav className="product-nav" aria-label="Vigil workspaces">
      <div className="product-nav-lockup">
        <Link className="product-nav-brand" href="/" aria-label="Vigil command center">
          <span aria-hidden="true"><i /></span>
          Vigil
        </Link>
        <small>Mercy Central / Waiting room 01</small>
      </div>
      <div className="product-nav-routes">
        {routes.map((route) => <Link className={current === route.id ? "active" : ""} href={route.href} aria-current={current === route.id ? "page" : undefined} key={route.id}>{route.label}</Link>)}
      </div>
      <span className="product-nav-context"><i /> System connected</span>
    </nav>
  );
}
