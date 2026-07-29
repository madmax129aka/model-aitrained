import React from "react";
import { NavLink } from "react-router-dom";

const navItems = [
  { to: "/", label: "Analyzer" },
  { to: "/batch", label: "Batch Mode" },
  { to: "/model-info", label: "Model Info" },
  { to: "/history", label: "History" },
  { to: "/about", label: "About" },
];

function NavItem({ to, label }) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `px-3 py-2 rounded-md text-sm font-medium transition-colors ${
          isActive
            ? "bg-forensic-cyan/10 text-forensic-cyan border border-forensic-cyan/30"
            : "text-slate-400 hover:text-slate-100 hover:bg-white/5 border border-transparent"
        }`
      }
    >
      {label}
    </NavLink>
  );
}

export default function Layout({ children }) {
  return (
    <div className="min-h-screen grid-texture bg-forensic-bg">
      <header className="border-b border-forensic-border/70 sticky top-0 z-30 bg-forensic-bg/90 backdrop-blur-md">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-3 flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-2.5">
            <div className="relative w-8 h-8 rounded-md bg-forensic-cyan/10 border border-forensic-cyan/40 flex items-center justify-center overflow-hidden">
              <span className="text-forensic-cyan font-mono font-bold text-sm">PT</span>
              <div className="absolute inset-x-0 h-1/3 bg-forensic-cyan/40 animate-scan-line" />
            </div>
            <div>
              <p className="font-mono font-bold text-slate-100 leading-tight tracking-wide">
                PIXEL<span className="text-forensic-cyan">TRUTH</span>
              </p>
              <p className="text-[10px] text-slate-500 leading-none tracking-wider uppercase">
                AI Image Detection Lab
              </p>
            </div>
          </div>
          <nav className="flex items-center gap-1 flex-wrap">
            {navItems.map((item) => (
              <NavItem key={item.to} {...item} />
            ))}
          </nav>
        </div>
      </header>
      <main className="max-w-6xl mx-auto px-4 sm:px-6 py-8">{children}</main>
      <footer className="border-t border-forensic-border/60 mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-6 text-xs text-slate-500 flex flex-wrap gap-2 justify-between">
          <span>
            Powered by a custom-trained CNN &mdash; no third-party AI-detection API.
          </span>
          <span>CIFAKE dataset &middot; 100,000 training images</span>
        </div>
      </footer>
    </div>
  );
}
