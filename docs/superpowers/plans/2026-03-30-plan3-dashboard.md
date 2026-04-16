# Plan 3: Next.js Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a production-quality Next.js dashboard with 7 pages (Overview, Signals, Portfolio, Backtest, Asset Detail, Audit Log, Settings) that connects directly to the existing PostgreSQL database via Prisma. Python runs as a background service only (writing data to the database); Next.js reads all data directly from PostgreSQL.

**Architecture:** Next.js App Router serves as both the UI and the API/backend layer. Prisma ORM connects directly to PostgreSQL for all database reads and writes. There is no proxying to the Python service -- Python is a background service that writes signals, portfolio data, backtest results, and audit logs directly to PostgreSQL. Next.js API routes query PostgreSQL via Prisma to serve all dashboard data. All data visualization uses Recharts with IDR-formatted amounts.

**Tech Stack:** Next.js 14+, TypeScript, Tailwind CSS, Prisma ORM, Recharts, React Hook Form, ESLint, Docker, GitHub Actions CI

---

## File Structure

```
services/nextjs/
├── Dockerfile
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── postcss.config.mjs
├── next.config.ts
├── .eslintrc.json
├── prisma/
│   └── schema.prisma
├── src/
│   ├── app/
│   │   ├── globals.css
│   │   ├── layout.tsx
│   │   ├── page.tsx                    # Overview
│   │   ├── signals/
│   │   │   └── page.tsx
│   │   ├── portfolio/
│   │   │   └── page.tsx
│   │   ├── backtest/
│   │   │   └── page.tsx
│   │   ├── assets/
│   │   │   └── [symbol]/
│   │   │       └── page.tsx
│   │   ├── audit/
│   │   │   └── page.tsx
│   │   ├── settings/
│   │   │   └── page.tsx
│   │   └── api/
│   │       ├── signals/
│   │       │   ├── route.ts
│   │       │   └── [id]/
│   │       │       └── route.ts
│   │       ├── portfolio/
│   │       │   └── route.ts
│   │       ├── backtest/
│   │       │   └── route.ts
│   │       ├── assets/
│   │       │   └── [symbol]/
│   │       │       └── route.ts
│   │       ├── audit/
│   │       │   └── route.ts
│   │       └── settings/
│   │           └── route.ts
│   ├── components/
│   │   ├── layout/
│   │   │   ├── sidebar.tsx
│   │   │   └── header.tsx
│   │   ├── charts/
│   │   │   ├── price-chart.tsx
│   │   │   └── performance-chart.tsx
│   │   ├── signals/
│   │   │   └── signal-card.tsx
│   │   └── portfolio/
│   │       └── position-row.tsx
│   ├── lib/
│   │   ├── prisma.ts
│   │   └── format.ts
│   └── types/
│       └── index.ts
└── __tests__/
    ├── lib/
    │   └── format.test.ts
    ├── components/
    │   ├── sidebar.test.tsx
    │   └── signal-card.test.tsx
    └── api/
        └── signals.test.ts
```

---

### Task 1: Next.js Project Scaffolding + Docker Integration

**Files:**
- Create: `services/nextjs/package.json`
- Create: `services/nextjs/tsconfig.json`
- Create: `services/nextjs/tailwind.config.ts`
- Create: `services/nextjs/postcss.config.mjs`
- Create: `services/nextjs/next.config.ts`
- Create: `services/nextjs/.eslintrc.json`
- Create: `services/nextjs/src/app/globals.css`
- Modify: `services/nextjs/Dockerfile`
- Modify: `docker-compose.yml`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-1-nextjs-scaffolding
```

- [ ] **Step 2: Create package.json**

Create `services/nextjs/package.json`:

```json
{
  "name": "ai-finance-dashboard",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint",
    "type-check": "tsc --noEmit",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:ci": "jest --ci --coverage",
    "prisma:generate": "prisma generate",
    "prisma:db:pull": "prisma db pull"
  },
  "dependencies": {
    "next": "^14.2.0",
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "@prisma/client": "^5.20.0",
    "recharts": "^2.12.0",
    "clsx": "^2.1.0",
    "tailwind-merge": "^2.5.0"
  },
  "devDependencies": {
    "typescript": "^5.5.0",
    "@types/node": "^20.14.0",
    "@types/react": "^18.3.0",
    "@types/react-dom": "^18.3.0",
    "tailwindcss": "^3.4.0",
    "postcss": "^8.4.0",
    "autoprefixer": "^10.4.0",
    "eslint": "^8.57.0",
    "eslint-config-next": "^14.2.0",
    "@typescript-eslint/eslint-plugin": "^7.0.0",
    "@typescript-eslint/parser": "^7.0.0",
    "prisma": "^5.20.0",
    "jest": "^29.7.0",
    "@jest/types": "^29.6.0",
    "ts-jest": "^29.2.0",
    "@testing-library/react": "^16.0.0",
    "@testing-library/jest-dom": "^6.5.0",
    "jest-environment-jsdom": "^29.7.0"
  }
}
```

- [ ] **Step 3: Create tsconfig.json**

Create `services/nextjs/tsconfig.json`:

```json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [
      {
        "name": "next"
      }
    ],
    "paths": {
      "@/*": ["./src/*"]
    },
    "baseUrl": "."
  },
  "include": ["next-env.d.ts", "**/*.ts", "**/*.tsx", ".next/types/**/*.ts"],
  "exclude": ["node_modules"]
}
```

- [ ] **Step 4: Create tailwind.config.ts**

Create `services/nextjs/tailwind.config.ts`:

```typescript
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#eff6ff",
          100: "#dbeafe",
          200: "#bfdbfe",
          300: "#93c5fd",
          400: "#60a5fa",
          500: "#3b82f6",
          600: "#2563eb",
          700: "#1d4ed8",
          800: "#1e40af",
          900: "#1e3a8a",
          950: "#172554",
        },
        signal: {
          green: "#22c55e",
          yellow: "#eab308",
          red: "#ef4444",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
    },
  },
  plugins: [],
};
export default config;
```

- [ ] **Step 5: Create postcss.config.mjs**

Create `services/nextjs/postcss.config.mjs`:

```javascript
/** @type {import('postcss-load-config').Config} */
const config = {
  plugins: {
    tailwindcss: {},
    autoprefixer: {},
  },
};

export default config;
```

- [ ] **Step 6: Create next.config.ts**

Create `services/nextjs/next.config.ts`:

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
};

export default nextConfig;
```

- [ ] **Step 7: Create .eslintrc.json**

Create `services/nextjs/.eslintrc.json`:

```json
{
  "extends": [
    "next/core-web-vitals",
    "plugin:@typescript-eslint/recommended"
  ],
  "parser": "@typescript-eslint/parser",
  "plugins": ["@typescript-eslint"],
  "rules": {
    "@typescript-eslint/no-unused-vars": ["error", { "argsIgnorePattern": "^_" }],
    "@typescript-eslint/no-explicit-any": "warn",
    "no-console": ["warn", { "allow": ["warn", "error"] }]
  }
}
```

- [ ] **Step 8: Create globals.css**

Create `services/nextjs/src/app/globals.css`:

```css
@tailwind base;
@tailwind components;
@tailwind utilities;

@layer base {
  :root {
    --background: #0f172a;
    --foreground: #f8fafc;
    --card: #1e293b;
    --card-foreground: #f1f5f9;
    --border: #334155;
    --muted: #475569;
    --muted-foreground: #94a3b8;
    --accent: #3b82f6;
    --accent-foreground: #ffffff;
    --destructive: #ef4444;
    --success: #22c55e;
    --warning: #eab308;
  }

  body {
    @apply bg-[var(--background)] text-[var(--foreground)] antialiased;
  }
}

@layer components {
  .card {
    @apply rounded-xl border border-[var(--border)] bg-[var(--card)] p-6 shadow-sm;
  }

  .badge-green {
    @apply inline-flex items-center rounded-full bg-green-500/10 px-2.5 py-0.5 text-xs font-medium text-green-400 ring-1 ring-inset ring-green-500/20;
  }

  .badge-yellow {
    @apply inline-flex items-center rounded-full bg-yellow-500/10 px-2.5 py-0.5 text-xs font-medium text-yellow-400 ring-1 ring-inset ring-yellow-500/20;
  }

  .badge-red {
    @apply inline-flex items-center rounded-full bg-red-500/10 px-2.5 py-0.5 text-xs font-medium text-red-400 ring-1 ring-inset ring-red-500/20;
  }

  .btn-primary {
    @apply rounded-lg bg-brand-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-brand-700 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 focus:ring-offset-slate-900 disabled:opacity-50 disabled:cursor-not-allowed;
  }

  .btn-secondary {
    @apply rounded-lg border border-[var(--border)] bg-transparent px-4 py-2 text-sm font-medium text-slate-300 transition-colors hover:bg-slate-800 focus:outline-none focus:ring-2 focus:ring-brand-500 focus:ring-offset-2 focus:ring-offset-slate-900;
  }

  .input-field {
    @apply w-full rounded-lg border border-[var(--border)] bg-slate-800 px-3 py-2 text-sm text-slate-100 placeholder-slate-500 focus:border-brand-500 focus:outline-none focus:ring-1 focus:ring-brand-500;
  }

  .table-header {
    @apply px-4 py-3 text-left text-xs font-medium uppercase tracking-wider text-slate-400;
  }

  .table-cell {
    @apply px-4 py-3 text-sm text-slate-300 whitespace-nowrap;
  }
}
```

- [ ] **Step 9: Create jest.config.ts**

Create `services/nextjs/jest.config.ts`:

```typescript
import type { Config } from "jest";
import nextJest from "next/jest";

const createJestConfig = nextJest({
  dir: "./",
});

const config: Config = {
  testEnvironment: "jsdom",
  setupFilesAfterSetup: ["<rootDir>/jest.setup.ts"],
  moduleNameMapper: {
    "^@/(.*)$": "<rootDir>/src/$1",
  },
  testPathPattern: ["<rootDir>/__tests__/"],
  collectCoverageFrom: [
    "src/**/*.{ts,tsx}",
    "!src/**/*.d.ts",
    "!src/app/api/**",
  ],
};

export default createJestConfig(config);
```

Create `services/nextjs/jest.setup.ts`:

```typescript
import "@testing-library/jest-dom";
```

- [ ] **Step 10: Update Dockerfile**

Replace `services/nextjs/Dockerfile` with:

```dockerfile
FROM node:20-alpine AS base

# --- Dependencies stage ---
FROM base AS deps
WORKDIR /app

COPY package.json ./
RUN npm install --frozen-lockfile 2>/dev/null || npm install

# --- Builder stage ---
FROM base AS builder
WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY . .

RUN npx prisma generate
RUN npm run build

# --- Runner stage ---
FROM base AS runner
WORKDIR /app

ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]
```

- [ ] **Step 11: Update docker-compose.yml**

Update the `nextjs` service in `docker-compose.yml` to:

```yaml
services:
  postgres:
    image: postgres:16-alpine
    env_file: .env
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-aifinance}"]
      interval: 5s
      timeout: 5s
      retries: 5

  python:
    build:
      context: ./services/python
      dockerfile: Dockerfile
    env_file: .env
    ports:
      - "8000:8000"
    volumes:
      - ./services/python/src:/app/src
      - ./services/python/tests:/app/tests
      - ./data/raw:/app/data/raw
    depends_on:
      postgres:
        condition: service_healthy

  nextjs:
    build:
      context: ./services/nextjs
      dockerfile: Dockerfile
    env_file: .env
    environment:
      - DATABASE_URL=postgresql://${POSTGRES_USER:-aifinance}:${POSTGRES_PASSWORD:-aifinance_dev}@postgres:5432/${POSTGRES_DB:-aifinance}
    ports:
      - "3000:3000"
    volumes:
      - ./services/nextjs/src:/app/src
      - ./services/nextjs/prisma:/app/prisma
    depends_on:
      postgres:
        condition: service_healthy

volumes:
  postgres_data:
```

- [ ] **Step 12: Update .env.example**

Append to `.env.example`:

```env
# Next.js
DATABASE_URL=postgresql://aifinance:aifinance_dev@postgres:5432/aifinance
```

- [ ] **Step 13: Update .gitignore**

Append to `.gitignore`:

```gitignore
# Next.js (additional)
services/nextjs/node_modules/
services/nextjs/.next/
services/nextjs/coverage/
services/nextjs/next-env.d.ts
```

- [ ] **Step 14: Verify scaffolding**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm install
npm run type-check
npm run lint
```

Expected: No TypeScript errors, no ESLint errors.

- [ ] **Step 15: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/package.json services/nextjs/tsconfig.json \
  services/nextjs/tailwind.config.ts services/nextjs/postcss.config.mjs \
  services/nextjs/next.config.ts services/nextjs/.eslintrc.json \
  services/nextjs/src/app/globals.css services/nextjs/Dockerfile \
  services/nextjs/jest.config.ts services/nextjs/jest.setup.ts \
  docker-compose.yml .env.example .gitignore
git commit -m "feat: scaffold Next.js project with TypeScript, Tailwind CSS, Docker integration"
git push -u origin feat/task-1-nextjs-scaffolding
gh pr create --title "feat: Next.js project scaffolding" --body "$(cat <<'EOF'
## Summary
- Scaffold Next.js 14 project with TypeScript, Tailwind CSS, ESLint
- Multi-stage Dockerfile with standalone output
- Update Docker Compose with DATABASE_URL
- Jest test configuration with jsdom environment

## Test plan
- [ ] `npm run type-check` passes
- [ ] `npm run lint` passes
- [ ] `docker compose build nextjs` succeeds
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 2: Prisma Schema + Database Client + Types

**Files:**
- Create: `services/nextjs/prisma/schema.prisma`
- Create: `services/nextjs/src/lib/prisma.ts`
- Create: `services/nextjs/src/lib/format.ts`
- Create: `services/nextjs/src/types/index.ts`
- Create: `services/nextjs/__tests__/lib/format.test.ts`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-2-prisma-types
```

- [ ] **Step 2: Create Prisma schema**

Create `services/nextjs/prisma/schema.prisma`:

```prisma
generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model AssetPriceHourly {
  id        Int      @id @default(autoincrement())
  asset     String   @db.VarChar(20)
  timestamp DateTime @db.Timestamptz()
  open      Float
  high      Float
  low       Float
  close     Float
  volume    Float

  @@unique([asset, timestamp], name: "uq_hourly_asset_ts")
  @@index([asset])
  @@index([timestamp])
  @@map("asset_prices_hourly")
}

model AssetPriceDaily {
  id     Int      @id @default(autoincrement())
  asset  String   @db.VarChar(20)
  date   DateTime @db.Timestamptz()
  open   Float
  high   Float
  low    Float
  close  Float
  volume Float

  @@unique([asset, date], name: "uq_daily_asset_date")
  @@index([asset])
  @@map("asset_prices_daily")
}

model AssetFundamental {
  id                 Int      @id @default(autoincrement())
  asset              String   @db.VarChar(20)
  fetched_at         DateTime @db.Timestamptz() @map("fetched_at")
  market_cap         Float?
  market_cap_rank    Int?     @map("market_cap_rank")
  total_volume_24h   Float?   @map("total_volume_24h")
  circulating_supply Float?   @map("circulating_supply")
  category           String?  @db.VarChar(100)

  @@index([asset])
  @@map("asset_fundamentals")
}

model Signal {
  id                 Int      @id @default(autoincrement())
  asset              String   @db.VarChar(20)
  action             String   @db.VarChar(10)
  confidence         Float
  suggested_hold_days Int     @map("suggested_hold_days")
  stop_loss_pct      Float    @map("stop_loss_pct")
  expected_return_pct Float   @map("expected_return_pct")
  model_agreement    String   @db.VarChar(10) @map("model_agreement")
  acknowledged       Boolean  @default(false)
  created_at         DateTime @db.Timestamptz() @map("created_at")

  @@index([asset])
  @@map("signals")
}

model Portfolio {
  id              Int       @id @default(autoincrement())
  asset           String    @db.VarChar(20)
  action          String    @db.VarChar(10)
  entry_price     Float     @map("entry_price")
  entry_amount_idr Int      @map("entry_amount_idr")
  quantity        Float
  stop_loss_price Float     @map("stop_loss_price")
  exit_price      Float?    @map("exit_price")
  exit_amount_idr Int?      @map("exit_amount_idr")
  pnl_idr         Int?      @map("pnl_idr")
  pnl_pct         Float?    @map("pnl_pct")
  status          String    @db.VarChar(10)
  opened_at       DateTime  @db.Timestamptz() @map("opened_at")
  closed_at       DateTime? @db.Timestamptz() @map("closed_at")
  signal_id       Int?      @map("signal_id")

  @@index([asset])
  @@map("portfolio")
}

model AuditLog {
  id         Int      @id @default(autoincrement())
  event_type String   @db.VarChar(50) @map("event_type")
  asset      String?  @db.VarChar(20)
  details    String?  @db.Text
  created_at DateTime @db.Timestamptz() @map("created_at")

  @@index([event_type])
  @@map("audit_log")
}
```

- [ ] **Step 3: Create Prisma client singleton**

Create `services/nextjs/src/lib/prisma.ts`:

```typescript
import { PrismaClient } from "@prisma/client";

const globalForPrisma = globalThis as unknown as {
  prisma: PrismaClient | undefined;
};

export const prisma =
  globalForPrisma.prisma ??
  new PrismaClient({
    log:
      process.env.NODE_ENV === "development"
        ? ["query", "error", "warn"]
        : ["error"],
  });

if (process.env.NODE_ENV !== "production") {
  globalForPrisma.prisma = prisma;
}
```

- [ ] **Step 4: Create TypeScript types**

Create `services/nextjs/src/types/index.ts`:

```typescript
// ----- Signal types -----

export type SignalAction = "BUY" | "SELL" | "HOLD" | "EXIT";

export type UrgencyLevel = "green" | "yellow" | "red";

export type AcknowledgeStatus = "seen" | "acted" | "skipped";

export interface SignalRecord {
  id: number;
  asset: string;
  action: SignalAction;
  confidence: number;
  suggested_hold_days: number;
  stop_loss_pct: number;
  expected_return_pct: number;
  model_agreement: string;
  acknowledged: boolean;
  created_at: string;
  urgency?: UrgencyLevel;
}

// ----- Portfolio types -----

export type PositionStatus = "open" | "closed" | "stopped";

export interface PortfolioPosition {
  id: number;
  asset: string;
  action: string;
  entry_price: number;
  entry_amount_idr: number;
  quantity: number;
  stop_loss_price: number;
  exit_price: number | null;
  exit_amount_idr: number | null;
  pnl_idr: number | null;
  pnl_pct: number | null;
  status: PositionStatus;
  opened_at: string;
  closed_at: string | null;
  signal_id: number | null;
  current_price?: number;
  unrealized_pnl_idr?: number;
  unrealized_pnl_pct?: number;
}

// ----- Price types -----

export interface PricePoint {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface DailyPrice {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ----- Backtest types -----

export interface BacktestMetrics {
  total_return_idr: number;
  total_return_pct: number;
  win_rate: number;
  reward_risk_ratio: number;
  sharpe_ratio: number;
  max_drawdown_pct: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
}

export interface MonthlyBreakdown {
  month: string;
  return_idr: number;
  return_pct: number;
  trades: number;
  win_rate: number;
}

export interface BacktestResult {
  metrics: BacktestMetrics;
  monthly: MonthlyBreakdown[];
  equity_curve: { date: string; value: number }[];
}

// ----- Asset detail types -----

export interface AssetDetail {
  symbol: string;
  current_price: number;
  price_history: DailyPrice[];
  fundamentals: {
    market_cap: number | null;
    market_cap_rank: number | null;
    total_volume_24h: number | null;
    circulating_supply: number | null;
    category: string | null;
  } | null;
  recent_signals: SignalRecord[];
  trade_history: PortfolioPosition[];
  model_scores: {
    xgboost: number | null;
    lightgbm: number | null;
    lstm: number | null;
    ensemble: number | null;
  };
  features: Record<string, number>;
}

// ----- Audit types -----

export interface AuditEntry {
  id: number;
  event_type: string;
  asset: string | null;
  details: string | null;
  created_at: string;
  parsed_details?: Record<string, unknown>;
}

// ----- Settings types -----

export interface RiskSettings {
  stop_loss_pct: number;
  max_positions: number;
  confidence_threshold: number;
  position_size_idr: number;
  max_single_asset_exposure_pct: number;
  portfolio_drawdown_pause_pct: number;
  retrain_schedule: string;
}

// ----- Overview types -----

export interface OverviewData {
  portfolio_value_idr: number;
  total_pnl_idr: number;
  total_pnl_pct: number;
  open_positions_count: number;
  max_positions: number;
  todays_signals: SignalRecord[];
  portfolio_history: { date: string; value: number }[];
}

// ----- API response wrapper -----

export interface ApiResponse<T> {
  data: T;
  error?: string;
}
```

- [ ] **Step 5: Create format utilities**

Create `services/nextjs/src/lib/format.ts`:

```typescript
/**
 * Format a number as Indonesian Rupiah (IDR).
 * Examples:
 *   formatIDR(1500000)    => "Rp 1.500.000"
 *   formatIDR(-250000)    => "-Rp 250.000"
 *   formatIDR(1234567890) => "Rp 1.234.567.890"
 */
export function formatIDR(amount: number): string {
  const isNegative = amount < 0;
  const absAmount = Math.abs(Math.round(amount));
  const formatted = absAmount.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ".");
  return `${isNegative ? "-" : ""}Rp ${formatted}`;
}

/**
 * Format a number as IDR with compact notation for large values.
 * Examples:
 *   formatIDRCompact(1500000)      => "Rp 1,5 jt"
 *   formatIDRCompact(1500000000)   => "Rp 1,5 M"
 *   formatIDRCompact(250000)       => "Rp 250.000"
 */
export function formatIDRCompact(amount: number): string {
  const absAmount = Math.abs(amount);
  const isNegative = amount < 0;
  const sign = isNegative ? "-" : "";

  if (absAmount >= 1_000_000_000) {
    const value = absAmount / 1_000_000_000;
    return `${sign}Rp ${value.toFixed(1).replace(".", ",")} M`;
  }
  if (absAmount >= 1_000_000) {
    const value = absAmount / 1_000_000;
    return `${sign}Rp ${value.toFixed(1).replace(".", ",")} jt`;
  }
  return formatIDR(amount);
}

/**
 * Format a percentage value with sign and fixed decimals.
 * Examples:
 *   formatPercent(12.5)  => "+12.50%"
 *   formatPercent(-8.3)  => "-8.30%"
 *   formatPercent(0)     => "0.00%"
 */
export function formatPercent(value: number, decimals: number = 2): string {
  const sign = value > 0 ? "+" : "";
  return `${sign}${value.toFixed(decimals)}%`;
}

/**
 * Format a USD price with appropriate decimal places.
 * Examples:
 *   formatUSD(70123.45)  => "$70,123.45"
 *   formatUSD(0.0034)    => "$0.0034"
 */
export function formatUSD(amount: number): string {
  if (amount < 0.01 && amount > 0) {
    return `$${amount.toPrecision(4)}`;
  }
  return `$${amount.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/**
 * Format a confidence score as a percentage.
 * Examples:
 *   formatConfidence(0.84) => "84%"
 *   formatConfidence(0.7)  => "70%"
 */
export function formatConfidence(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/**
 * Format a datetime string to locale display.
 * Examples:
 *   formatDateTime("2026-03-30T00:15:00Z") => "30 Mar 2026, 07:15"
 */
export function formatDateTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false,
  });
}

/**
 * Format a date string to short display.
 * Examples:
 *   formatDate("2026-03-30") => "30 Mar 2026"
 */
export function formatDate(dateString: string): string {
  const date = new Date(dateString);
  return date.toLocaleDateString("id-ID", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/**
 * Determine the urgency level for a signal based on its properties.
 */
export function getUrgencyLevel(signal: {
  action: string;
  confidence: number;
  stop_loss_pct: number;
}): "green" | "yellow" | "red" {
  if (signal.action === "EXIT" || signal.action === "SELL") {
    return "red";
  }
  if (signal.confidence < 0.6) {
    return "yellow";
  }
  if (signal.confidence < 0.5) {
    return "red";
  }
  return "green";
}

/**
 * Get a CSS class name for P&L coloring.
 */
export function getPnlColor(value: number): string {
  if (value > 0) return "text-green-400";
  if (value < 0) return "text-red-400";
  return "text-slate-400";
}

/**
 * Get a CSS class name for urgency badge.
 */
export function getUrgencyBadgeClass(urgency: "green" | "yellow" | "red"): string {
  switch (urgency) {
    case "green":
      return "badge-green";
    case "yellow":
      return "badge-yellow";
    case "red":
      return "badge-red";
  }
}
```

- [ ] **Step 6: Write format utility tests**

Create `services/nextjs/__tests__/lib/format.test.ts`:

```typescript
import {
  formatIDR,
  formatIDRCompact,
  formatPercent,
  formatUSD,
  formatConfidence,
  getUrgencyLevel,
  getPnlColor,
  getUrgencyBadgeClass,
} from "@/lib/format";

describe("formatIDR", () => {
  it("formats positive amounts", () => {
    expect(formatIDR(1500000)).toBe("Rp 1.500.000");
    expect(formatIDR(1000000)).toBe("Rp 1.000.000");
    expect(formatIDR(0)).toBe("Rp 0");
    expect(formatIDR(1234567890)).toBe("Rp 1.234.567.890");
  });

  it("formats negative amounts", () => {
    expect(formatIDR(-250000)).toBe("-Rp 250.000");
    expect(formatIDR(-1500000)).toBe("-Rp 1.500.000");
  });

  it("rounds to nearest integer", () => {
    expect(formatIDR(1500000.7)).toBe("Rp 1.500.001");
    expect(formatIDR(1500000.3)).toBe("Rp 1.500.000");
  });
});

describe("formatIDRCompact", () => {
  it("formats millions as jt", () => {
    expect(formatIDRCompact(1500000)).toBe("Rp 1,5 jt");
    expect(formatIDRCompact(8000000)).toBe("Rp 8,0 jt");
  });

  it("formats billions as M", () => {
    expect(formatIDRCompact(1500000000)).toBe("Rp 1,5 M");
  });

  it("falls back to full format for small values", () => {
    expect(formatIDRCompact(250000)).toBe("Rp 250.000");
  });

  it("handles negative compact values", () => {
    expect(formatIDRCompact(-2500000)).toBe("-Rp 2,5 jt");
  });
});

describe("formatPercent", () => {
  it("adds plus sign for positive values", () => {
    expect(formatPercent(12.5)).toBe("+12.50%");
  });

  it("shows minus sign for negative values", () => {
    expect(formatPercent(-8.3)).toBe("-8.30%");
  });

  it("shows zero without sign", () => {
    expect(formatPercent(0)).toBe("0.00%");
  });

  it("respects custom decimal places", () => {
    expect(formatPercent(12.5, 1)).toBe("+12.5%");
  });
});

describe("formatUSD", () => {
  it("formats standard prices", () => {
    expect(formatUSD(70123.45)).toBe("$70,123.45");
  });

  it("formats small prices with precision", () => {
    expect(formatUSD(0.0034)).toBe("$0.003400");
  });
});

describe("formatConfidence", () => {
  it("converts decimal to percentage", () => {
    expect(formatConfidence(0.84)).toBe("84%");
    expect(formatConfidence(0.7)).toBe("70%");
    expect(formatConfidence(1.0)).toBe("100%");
  });
});

describe("getUrgencyLevel", () => {
  it("returns red for EXIT action", () => {
    expect(
      getUrgencyLevel({ action: "EXIT", confidence: 0.9, stop_loss_pct: -8 })
    ).toBe("red");
  });

  it("returns red for SELL action", () => {
    expect(
      getUrgencyLevel({ action: "SELL", confidence: 0.8, stop_loss_pct: -8 })
    ).toBe("red");
  });

  it("returns yellow for low confidence BUY", () => {
    expect(
      getUrgencyLevel({ action: "BUY", confidence: 0.55, stop_loss_pct: -8 })
    ).toBe("yellow");
  });

  it("returns green for high confidence BUY", () => {
    expect(
      getUrgencyLevel({ action: "BUY", confidence: 0.84, stop_loss_pct: -8 })
    ).toBe("green");
  });
});

describe("getPnlColor", () => {
  it("returns green for positive P&L", () => {
    expect(getPnlColor(100000)).toBe("text-green-400");
  });

  it("returns red for negative P&L", () => {
    expect(getPnlColor(-50000)).toBe("text-red-400");
  });

  it("returns slate for zero P&L", () => {
    expect(getPnlColor(0)).toBe("text-slate-400");
  });
});

describe("getUrgencyBadgeClass", () => {
  it("returns correct badge classes", () => {
    expect(getUrgencyBadgeClass("green")).toBe("badge-green");
    expect(getUrgencyBadgeClass("yellow")).toBe("badge-yellow");
    expect(getUrgencyBadgeClass("red")).toBe("badge-red");
  });
});
```

- [ ] **Step 7: Generate Prisma client and run tests**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npx prisma generate
npm test -- --passWithNoTests
npm test -- __tests__/lib/format.test.ts
```

Expected: All format tests pass.

- [ ] **Step 8: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/prisma/schema.prisma \
  services/nextjs/src/lib/prisma.ts \
  services/nextjs/src/lib/format.ts \
  services/nextjs/src/types/index.ts \
  services/nextjs/__tests__/lib/format.test.ts
git commit -m "feat: add Prisma schema, types, and format utilities"
git push -u origin feat/task-2-prisma-types
gh pr create --title "feat: Prisma schema + types + utilities" --body "$(cat <<'EOF'
## Summary
- Prisma schema mapping all 6 PostgreSQL tables from Plan 1
- TypeScript type definitions for all domain entities
- IDR/USD/percentage formatting utilities with full test coverage

## Test plan
- [ ] `npx prisma generate` succeeds
- [ ] `npm test -- __tests__/lib/format.test.ts` all pass
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 3: Layout Components (Sidebar + Header)

**Files:**
- Create: `services/nextjs/src/components/layout/sidebar.tsx`
- Create: `services/nextjs/src/components/layout/header.tsx`
- Create: `services/nextjs/src/app/layout.tsx`
- Create: `services/nextjs/__tests__/components/sidebar.test.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-3-layout-components
```

- [ ] **Step 2: Create Sidebar component**

Create `services/nextjs/src/components/layout/sidebar.tsx`:

```typescript
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { clsx } from "clsx";

interface NavItem {
  name: string;
  href: string;
  icon: React.ReactNode;
}

const navigation: NavItem[] = [
  {
    name: "Overview",
    href: "/",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6A2.25 2.25 0 0 1 6 3.75h2.25A2.25 2.25 0 0 1 10.5 6v2.25a2.25 2.25 0 0 1-2.25 2.25H6a2.25 2.25 0 0 1-2.25-2.25V6ZM3.75 15.75A2.25 2.25 0 0 1 6 13.5h2.25a2.25 2.25 0 0 1 2.25 2.25V18a2.25 2.25 0 0 1-2.25 2.25H6A2.25 2.25 0 0 1 3.75 18v-2.25ZM13.5 6a2.25 2.25 0 0 1 2.25-2.25H18A2.25 2.25 0 0 1 20.25 6v2.25A2.25 2.25 0 0 1 18 10.5h-2.25a2.25 2.25 0 0 1-2.25-2.25V6ZM13.5 15.75a2.25 2.25 0 0 1 2.25-2.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 20.25h-2.25A2.25 2.25 0 0 1 13.5 18v-2.25Z" />
      </svg>
    ),
  },
  {
    name: "Signals",
    href: "/signals",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M14.857 17.082a23.848 23.848 0 0 0 5.454-1.31A8.967 8.967 0 0 1 18 9.75V9A6 6 0 0 0 6 9v.75a8.967 8.967 0 0 1-2.312 6.022c1.733.64 3.56 1.085 5.455 1.31m5.714 0a24.255 24.255 0 0 1-5.714 0m5.714 0a3 3 0 1 1-5.714 0" />
      </svg>
    ),
  },
  {
    name: "Portfolio",
    href: "/portfolio",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M2.25 18.75a60.07 60.07 0 0 1 15.797 2.101c.727.198 1.453-.342 1.453-1.096V18.75M3.75 4.5v.75A.75.75 0 0 1 3 6h-.75m0 0v-.375c0-.621.504-1.125 1.125-1.125H20.25M2.25 6v9m18-10.5v.75c0 .414.336.75.75.75h.75m-1.5-1.5h.375c.621 0 1.125.504 1.125 1.125v9.75c0 .621-.504 1.125-1.125 1.125h-.375m1.5-1.5H21a.75.75 0 0 0-.75.75v.75m0 0H3.75m0 0h-.375a1.125 1.125 0 0 1-1.125-1.125V15m1.5 1.5v-.75A.75.75 0 0 0 3 15h-.75M15 10.5a3 3 0 1 1-6 0 3 3 0 0 1 6 0Zm3 0h.008v.008H18V10.5Zm-12 0h.008v.008H6V10.5Z" />
      </svg>
    ),
  },
  {
    name: "Backtest",
    href: "/backtest",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M3 13.125C3 12.504 3.504 12 4.125 12h2.25c.621 0 1.125.504 1.125 1.125v6.75C7.5 20.496 6.996 21 6.375 21h-2.25A1.125 1.125 0 0 1 3 19.875v-6.75ZM9.75 8.625c0-.621.504-1.125 1.125-1.125h2.25c.621 0 1.125.504 1.125 1.125v11.25c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V8.625ZM16.5 4.125c0-.621.504-1.125 1.125-1.125h2.25C20.496 3 21 3.504 21 4.125v15.75c0 .621-.504 1.125-1.125 1.125h-2.25a1.125 1.125 0 0 1-1.125-1.125V4.125Z" />
      </svg>
    ),
  },
  {
    name: "Audit Log",
    href: "/audit",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 0 0-3.375-3.375h-1.5A1.125 1.125 0 0 1 13.5 7.125v-1.5a3.375 3.375 0 0 0-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 0 0-9-9Z" />
      </svg>
    ),
  },
  {
    name: "Settings",
    href: "/settings",
    icon: (
      <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
        <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
        <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
      </svg>
    ),
  },
];

export function Sidebar() {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile overlay sidebar */}
      <div className="lg:hidden">
        <MobileSidebar pathname={pathname} />
      </div>

      {/* Desktop sidebar */}
      <div className="hidden lg:fixed lg:inset-y-0 lg:z-50 lg:flex lg:w-64 lg:flex-col">
        <div className="flex grow flex-col gap-y-5 overflow-y-auto border-r border-[var(--border)] bg-[var(--card)] px-6 pb-4">
          <div className="flex h-16 shrink-0 items-center">
            <span className="text-xl font-bold text-brand-400">AI-Finance</span>
          </div>
          <nav className="flex flex-1 flex-col">
            <ul className="flex flex-1 flex-col gap-y-1">
              {navigation.map((item) => {
                const isActive =
                  item.href === "/"
                    ? pathname === "/"
                    : pathname.startsWith(item.href);
                return (
                  <li key={item.name}>
                    <Link
                      href={item.href}
                      className={clsx(
                        "group flex gap-x-3 rounded-md px-3 py-2 text-sm font-medium leading-6 transition-colors",
                        isActive
                          ? "bg-brand-600/10 text-brand-400"
                          : "text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                      )}
                    >
                      <span
                        className={clsx(
                          "shrink-0",
                          isActive ? "text-brand-400" : "text-slate-500 group-hover:text-slate-300"
                        )}
                      >
                        {item.icon}
                      </span>
                      {item.name}
                    </Link>
                  </li>
                );
              })}
            </ul>
          </nav>
          <div className="border-t border-[var(--border)] pt-4">
            <p className="text-xs text-slate-500">AI-Finance v0.1.0</p>
            <p className="text-xs text-slate-600">Local Trading System</p>
          </div>
        </div>
      </div>
    </>
  );
}

function MobileSidebar({ pathname }: { pathname: string }) {
  return (
    <div className="fixed bottom-0 left-0 right-0 z-50 border-t border-[var(--border)] bg-[var(--card)]">
      <nav className="flex justify-around px-2 py-2">
        {navigation.slice(0, 5).map((item) => {
          const isActive =
            item.href === "/"
              ? pathname === "/"
              : pathname.startsWith(item.href);
          return (
            <Link
              key={item.name}
              href={item.href}
              className={clsx(
                "flex flex-col items-center gap-1 rounded-md px-2 py-1 text-xs transition-colors",
                isActive
                  ? "text-brand-400"
                  : "text-slate-500 hover:text-slate-300"
              )}
            >
              {item.icon}
              <span>{item.name}</span>
            </Link>
          );
        })}
      </nav>
    </div>
  );
}
```

- [ ] **Step 3: Create Header component**

Create `services/nextjs/src/components/layout/header.tsx`:

```typescript
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const pageTitles: Record<string, string> = {
  "/": "Overview",
  "/signals": "Signals",
  "/portfolio": "Portfolio",
  "/backtest": "Backtest",
  "/audit": "Audit Log",
  "/settings": "Settings",
};

export function Header() {
  const pathname = usePathname();

  const getPageTitle = (): string => {
    if (pageTitles[pathname]) return pageTitles[pathname];
    if (pathname.startsWith("/assets/")) {
      const symbol = pathname.split("/")[2]?.toUpperCase() || "Asset";
      return `${symbol} Detail`;
    }
    return "AI-Finance";
  };

  return (
    <header className="sticky top-0 z-40 flex h-16 shrink-0 items-center gap-x-4 border-b border-[var(--border)] bg-[var(--background)]/80 px-4 backdrop-blur-sm sm:gap-x-6 sm:px-6 lg:px-8">
      <div className="flex flex-1 items-center gap-x-4 self-stretch lg:gap-x-6">
        <div className="flex flex-1 items-center">
          <h1 className="text-lg font-semibold text-slate-100">
            {getPageTitle()}
          </h1>
        </div>
        <div className="flex items-center gap-x-4 lg:gap-x-6">
          <Link
            href="/settings"
            className="rounded-md p-2 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200"
            aria-label="Settings"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" strokeWidth={1.5} stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9.594 3.94c.09-.542.56-.94 1.11-.94h2.593c.55 0 1.02.398 1.11.94l.213 1.281c.063.374.313.686.645.87.074.04.147.083.22.127.325.196.72.257 1.075.124l1.217-.456a1.125 1.125 0 0 1 1.37.49l1.296 2.247a1.125 1.125 0 0 1-.26 1.431l-1.003.827c-.293.241-.438.613-.43.992a7.723 7.723 0 0 1 0 .255c-.008.378.137.75.43.991l1.004.827c.424.35.534.955.26 1.43l-1.298 2.247a1.125 1.125 0 0 1-1.369.491l-1.217-.456c-.355-.133-.75-.072-1.076.124a6.47 6.47 0 0 1-.22.128c-.331.183-.581.495-.644.869l-.213 1.281c-.09.543-.56.94-1.11.94h-2.594c-.55 0-1.019-.398-1.11-.94l-.213-1.281c-.062-.374-.312-.686-.644-.87a6.52 6.52 0 0 1-.22-.127c-.325-.196-.72-.257-1.076-.124l-1.217.456a1.125 1.125 0 0 1-1.369-.49l-1.297-2.247a1.125 1.125 0 0 1 .26-1.431l1.004-.827c.292-.24.437-.613.43-.991a6.932 6.932 0 0 1 0-.255c.007-.38-.138-.751-.43-.992l-1.004-.827a1.125 1.125 0 0 1-.26-1.43l1.297-2.247a1.125 1.125 0 0 1 1.37-.491l1.216.456c.356.133.751.072 1.076-.124.072-.044.146-.086.22-.128.332-.183.582-.495.644-.869l.214-1.28Z" />
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 12a3 3 0 1 1-6 0 3 3 0 0 1 6 0Z" />
            </svg>
          </Link>
        </div>
      </div>
    </header>
  );
}

```

- [ ] **Step 4: Create root layout**

Create `services/nextjs/src/app/layout.tsx`:

```typescript
import type { Metadata } from "next";
import { Sidebar } from "@/components/layout/sidebar";
import { Header } from "@/components/layout/header";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI-Finance Dashboard",
  description: "Crypto trading system with ML-driven signals",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full">
        <Sidebar />
        <div className="lg:pl-64">
          <Header />
          <main className="px-4 py-6 sm:px-6 lg:px-8 pb-20 lg:pb-6">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
```

- [ ] **Step 5: Write sidebar test**

Create `services/nextjs/__tests__/components/sidebar.test.tsx`:

```typescript
import { render, screen } from "@testing-library/react";
import { Sidebar } from "@/components/layout/sidebar";

// Mock next/navigation
jest.mock("next/navigation", () => ({
  usePathname: () => "/",
}));

describe("Sidebar", () => {
  it("renders all navigation items", () => {
    render(<Sidebar />);
    expect(screen.getAllByText("Overview").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Signals").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Portfolio").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Backtest").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Audit Log").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Settings").length).toBeGreaterThan(0);
  });

  it("renders the brand name", () => {
    render(<Sidebar />);
    expect(screen.getByText("AI-Finance")).toBeInTheDocument();
  });

  it("highlights the active route", () => {
    render(<Sidebar />);
    const overviewLinks = screen.getAllByRole("link", { name: /overview/i });
    const hasActiveClass = overviewLinks.some((link) =>
      link.className.includes("text-brand-400")
    );
    expect(hasActiveClass).toBe(true);
  });
});
```

- [ ] **Step 6: Run tests**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm test -- __tests__/components/sidebar.test.tsx
```

Expected: All tests pass.

- [ ] **Step 7: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/components/layout/sidebar.tsx \
  services/nextjs/src/components/layout/header.tsx \
  services/nextjs/src/app/layout.tsx \
  services/nextjs/__tests__/components/sidebar.test.tsx
git commit -m "feat: add sidebar navigation, header, and root layout"
git push -u origin feat/task-3-layout-components
gh pr create --title "feat: layout components (sidebar + header)" --body "$(cat <<'EOF'
## Summary
- Responsive sidebar with 6 navigation items and active state highlighting
- Mobile bottom navigation bar for small screens
- Header with dynamic page title and settings link
- Root layout composing sidebar + header + content area

## Test plan
- [ ] Sidebar renders all nav items
- [ ] Active route highlighting works
- [ ] Mobile navigation renders on small screens
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 4: Chart Components + Signal Card

**Files:**
- Create: `services/nextjs/src/components/charts/price-chart.tsx`
- Create: `services/nextjs/src/components/charts/performance-chart.tsx`
- Create: `services/nextjs/src/components/signals/signal-card.tsx`
- Create: `services/nextjs/src/components/portfolio/position-row.tsx`
- Create: `services/nextjs/__tests__/components/signal-card.test.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-4-chart-signal-components
```

- [ ] **Step 2: Create PriceChart component**

Create `services/nextjs/src/components/charts/price-chart.tsx`:

```typescript
"use client";

import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import type { DailyPrice } from "@/types";
import { formatUSD } from "@/lib/format";

interface PriceChartProps {
  data: DailyPrice[];
  height?: number;
  showVolume?: boolean;
}

export function PriceChart({ data, height = 400, showVolume = true }: PriceChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-slate-800/50 p-8" style={{ height }}>
        <p className="text-sm text-slate-500">No price data available</p>
      </div>
    );
  }

  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    close: d.close,
    high: d.high,
    low: d.low,
    volume: d.volume,
  }));

  return (
    <ResponsiveContainer width="100%" height={height}>
      <ComposedChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
        />
        <YAxis
          yAxisId="price"
          orientation="right"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
          tickFormatter={(value: number) => formatUSD(value)}
          domain={["auto", "auto"]}
        />
        {showVolume && (
          <YAxis
            yAxisId="volume"
            orientation="left"
            tick={{ fill: "#94a3b8", fontSize: 10 }}
            tickLine={{ stroke: "#475569" }}
            axisLine={{ stroke: "#475569" }}
            tickFormatter={(value: number) => {
              if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M`;
              if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K`;
              return value.toString();
            }}
          />
        )}
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(value: number, name: string) => {
            if (name === "volume") return [value.toLocaleString(), "Volume"];
            return [formatUSD(value), name.charAt(0).toUpperCase() + name.slice(1)];
          }}
        />
        <Legend
          wrapperStyle={{ color: "#94a3b8", fontSize: 12, paddingTop: 8 }}
        />
        {showVolume && (
          <Bar
            yAxisId="volume"
            dataKey="volume"
            fill="#3b82f6"
            opacity={0.15}
            name="volume"
          />
        )}
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="close"
          stroke="#3b82f6"
          strokeWidth={2}
          dot={false}
          name="close"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="high"
          stroke="#22c55e"
          strokeWidth={1}
          strokeDasharray="3 3"
          dot={false}
          name="high"
        />
        <Line
          yAxisId="price"
          type="monotone"
          dataKey="low"
          stroke="#ef4444"
          strokeWidth={1}
          strokeDasharray="3 3"
          dot={false}
          name="low"
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 3: Create PerformanceChart component**

Create `services/nextjs/src/components/charts/performance-chart.tsx`:

```typescript
"use client";

import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { formatIDRCompact } from "@/lib/format";

interface PerformanceChartProps {
  data: { date: string; value: number }[];
  height?: number;
  startingValue?: number;
}

export function PerformanceChart({
  data,
  height = 350,
  startingValue,
}: PerformanceChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border border-[var(--border)] bg-slate-800/50 p-8" style={{ height }}>
        <p className="text-sm text-slate-500">No performance data available</p>
      </div>
    );
  }

  const chartData = data.map((d) => ({
    date: new Date(d.date).toLocaleDateString("en-US", { month: "short", day: "numeric" }),
    value: d.value,
  }));

  const lastValue = chartData[chartData.length - 1]?.value ?? 0;
  const firstValue = startingValue ?? chartData[0]?.value ?? 0;
  const isPositive = lastValue >= firstValue;

  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <defs>
          <linearGradient id="performanceGradient" x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="5%"
              stopColor={isPositive ? "#22c55e" : "#ef4444"}
              stopOpacity={0.3}
            />
            <stop
              offset="95%"
              stopColor={isPositive ? "#22c55e" : "#ef4444"}
              stopOpacity={0}
            />
          </linearGradient>
        </defs>
        <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
        <XAxis
          dataKey="date"
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
        />
        <YAxis
          tick={{ fill: "#94a3b8", fontSize: 12 }}
          tickLine={{ stroke: "#475569" }}
          axisLine={{ stroke: "#475569" }}
          tickFormatter={(value: number) => formatIDRCompact(value)}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: "#1e293b",
            border: "1px solid #334155",
            borderRadius: "8px",
            color: "#f1f5f9",
          }}
          labelStyle={{ color: "#94a3b8" }}
          formatter={(value: number) => [formatIDRCompact(value), "Value"]}
        />
        {startingValue && (
          <ReferenceLine
            y={startingValue}
            stroke="#475569"
            strokeDasharray="5 5"
            label={{
              value: "Start",
              fill: "#94a3b8",
              fontSize: 11,
              position: "left",
            }}
          />
        )}
        <Area
          type="monotone"
          dataKey="value"
          stroke={isPositive ? "#22c55e" : "#ef4444"}
          strokeWidth={2}
          fill="url(#performanceGradient)"
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}
```

- [ ] **Step 4: Create SignalCard component**

Create `services/nextjs/src/components/signals/signal-card.tsx`:

```typescript
"use client";

import { clsx } from "clsx";
import type { SignalRecord, AcknowledgeStatus } from "@/types";
import {
  formatConfidence,
  formatPercent,
  formatDateTime,
  getUrgencyLevel,
  getUrgencyBadgeClass,
} from "@/lib/format";

interface SignalCardProps {
  signal: SignalRecord;
  onAcknowledge: (id: number, status: AcknowledgeStatus) => void;
  isLoading?: boolean;
}

export function SignalCard({ signal, onAcknowledge, isLoading = false }: SignalCardProps) {
  const urgency = signal.urgency ?? getUrgencyLevel(signal);

  const actionColors: Record<string, string> = {
    BUY: "text-green-400 bg-green-500/10 border-green-500/30",
    SELL: "text-red-400 bg-red-500/10 border-red-500/30",
    HOLD: "text-yellow-400 bg-yellow-500/10 border-yellow-500/30",
    EXIT: "text-red-400 bg-red-500/10 border-red-500/30",
  };

  return (
    <div
      className={clsx(
        "card relative overflow-hidden transition-shadow hover:shadow-md",
        signal.acknowledged && "opacity-60"
      )}
    >
      {/* Urgency indicator strip */}
      <div
        className={clsx(
          "absolute left-0 top-0 h-full w-1",
          urgency === "green" && "bg-green-500",
          urgency === "yellow" && "bg-yellow-500",
          urgency === "red" && "bg-red-500"
        )}
      />

      <div className="pl-4">
        {/* Header */}
        <div className="flex items-start justify-between">
          <div className="flex items-center gap-3">
            <span className="text-lg font-bold text-slate-100">{signal.asset}</span>
            <span
              className={clsx(
                "rounded-md border px-2 py-0.5 text-xs font-semibold uppercase",
                actionColors[signal.action] || "text-slate-400 bg-slate-500/10 border-slate-500/30"
              )}
            >
              {signal.action}
            </span>
            <span className={getUrgencyBadgeClass(urgency)}>
              {urgency.toUpperCase()}
            </span>
          </div>
          <span className="text-xs text-slate-500">
            {formatDateTime(signal.created_at)}
          </span>
        </div>

        {/* Metrics grid */}
        <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4">
          <div>
            <p className="text-xs text-slate-500">Confidence</p>
            <p className="text-sm font-medium text-slate-200">
              {formatConfidence(signal.confidence)}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Expected Return</p>
            <p
              className={clsx(
                "text-sm font-medium",
                signal.expected_return_pct > 0 ? "text-green-400" : "text-red-400"
              )}
            >
              {formatPercent(signal.expected_return_pct)}
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Hold Duration</p>
            <p className="text-sm font-medium text-slate-200">
              {signal.suggested_hold_days} days
            </p>
          </div>
          <div>
            <p className="text-xs text-slate-500">Model Agreement</p>
            <p className="text-sm font-medium text-slate-200">
              {signal.model_agreement}
            </p>
          </div>
        </div>

        {/* Stop loss */}
        <div className="mt-3">
          <p className="text-xs text-slate-500">
            Stop Loss: <span className="text-red-400">{formatPercent(signal.stop_loss_pct)}</span>
          </p>
        </div>

        {/* Acknowledge buttons */}
        {!signal.acknowledged && (
          <div className="mt-4 flex gap-2">
            <button
              onClick={() => onAcknowledge(signal.id, "acted")}
              disabled={isLoading}
              className="btn-primary text-xs"
            >
              Act
            </button>
            <button
              onClick={() => onAcknowledge(signal.id, "seen")}
              disabled={isLoading}
              className="btn-secondary text-xs"
            >
              Seen
            </button>
            <button
              onClick={() => onAcknowledge(signal.id, "skipped")}
              disabled={isLoading}
              className="btn-secondary text-xs"
            >
              Skip
            </button>
          </div>
        )}

        {signal.acknowledged && (
          <div className="mt-4">
            <span className="text-xs text-slate-500">Acknowledged</span>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Create PositionRow component**

Create `services/nextjs/src/components/portfolio/position-row.tsx`:

```typescript
"use client";

import Link from "next/link";
import { clsx } from "clsx";
import type { PortfolioPosition } from "@/types";
import {
  formatIDR,
  formatPercent,
  formatUSD,
  formatDateTime,
  getPnlColor,
} from "@/lib/format";

interface PositionRowProps {
  position: PortfolioPosition;
}

export function PositionRow({ position }: PositionRowProps) {
  const currentPrice = position.current_price ?? position.entry_price;
  const unrealizedPnlIdr =
    position.unrealized_pnl_idr ??
    (position.status === "open"
      ? Math.round((currentPrice - position.entry_price) * position.quantity * 16000)
      : null);
  const unrealizedPnlPct =
    position.unrealized_pnl_pct ??
    (position.status === "open"
      ? ((currentPrice - position.entry_price) / position.entry_price) * 100
      : null);

  const displayPnlIdr =
    position.status === "open" ? unrealizedPnlIdr : position.pnl_idr;
  const displayPnlPct =
    position.status === "open" ? unrealizedPnlPct : position.pnl_pct;

  const stopLossHitPct =
    ((currentPrice - position.stop_loss_price) /
      (position.entry_price - position.stop_loss_price)) *
    100;
  const stopLossProximity = Math.max(0, Math.min(100, 100 - stopLossHitPct));

  return (
    <tr className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50">
      <td className="table-cell">
        <Link
          href={`/assets/${position.asset.toLowerCase()}`}
          className="font-medium text-brand-400 hover:text-brand-300"
        >
          {position.asset}
        </Link>
      </td>
      <td className="table-cell">
        <span
          className={clsx(
            "rounded-md px-2 py-0.5 text-xs font-semibold uppercase",
            position.status === "open" && "bg-green-500/10 text-green-400",
            position.status === "closed" && "bg-slate-500/10 text-slate-400",
            position.status === "stopped" && "bg-red-500/10 text-red-400"
          )}
        >
          {position.status}
        </span>
      </td>
      <td className="table-cell font-mono">{formatUSD(position.entry_price)}</td>
      <td className="table-cell font-mono">{formatUSD(currentPrice)}</td>
      <td className="table-cell">{formatIDR(position.entry_amount_idr)}</td>
      <td className="table-cell">
        {displayPnlIdr !== null && displayPnlIdr !== undefined ? (
          <span className={clsx("font-mono", getPnlColor(displayPnlIdr))}>
            {formatIDR(displayPnlIdr)}
          </span>
        ) : (
          <span className="text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell">
        {displayPnlPct !== null && displayPnlPct !== undefined ? (
          <span className={clsx("font-mono", getPnlColor(displayPnlPct))}>
            {formatPercent(displayPnlPct)}
          </span>
        ) : (
          <span className="text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell">
        {position.status === "open" ? (
          <div className="flex items-center gap-2">
            <div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-700">
              <div
                className={clsx(
                  "h-full rounded-full transition-all",
                  stopLossProximity > 60 ? "bg-red-500" : stopLossProximity > 30 ? "bg-yellow-500" : "bg-green-500"
                )}
                style={{ width: `${stopLossProximity}%` }}
              />
            </div>
            <span className="text-xs text-slate-500 font-mono">
              {formatUSD(position.stop_loss_price)}
            </span>
          </div>
        ) : (
          <span className="text-xs text-slate-500">-</span>
        )}
      </td>
      <td className="table-cell text-xs text-slate-500">
        {formatDateTime(position.opened_at)}
      </td>
    </tr>
  );
}
```

- [ ] **Step 6: Write SignalCard test**

Create `services/nextjs/__tests__/components/signal-card.test.tsx`:

```typescript
import { render, screen, fireEvent } from "@testing-library/react";
import { SignalCard } from "@/components/signals/signal-card";
import type { SignalRecord } from "@/types";

const mockSignal: SignalRecord = {
  id: 1,
  asset: "ETH",
  action: "BUY",
  confidence: 0.84,
  suggested_hold_days: 18,
  stop_loss_pct: -8,
  expected_return_pct: 12,
  model_agreement: "3/3",
  acknowledged: false,
  created_at: "2026-03-30T00:15:00Z",
};

describe("SignalCard", () => {
  it("renders signal details", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);

    expect(screen.getByText("ETH")).toBeInTheDocument();
    expect(screen.getByText("BUY")).toBeInTheDocument();
    expect(screen.getByText("84%")).toBeInTheDocument();
    expect(screen.getByText("+12.00%")).toBeInTheDocument();
    expect(screen.getByText("18 days")).toBeInTheDocument();
    expect(screen.getByText("3/3")).toBeInTheDocument();
  });

  it("shows acknowledge buttons when not acknowledged", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);

    expect(screen.getByText("Act")).toBeInTheDocument();
    expect(screen.getByText("Seen")).toBeInTheDocument();
    expect(screen.getByText("Skip")).toBeInTheDocument();
  });

  it("hides acknowledge buttons when acknowledged", () => {
    const acknowledged = { ...mockSignal, acknowledged: true };
    render(<SignalCard signal={acknowledged} onAcknowledge={() => {}} />);

    expect(screen.queryByText("Act")).not.toBeInTheDocument();
    expect(screen.getByText("Acknowledged")).toBeInTheDocument();
  });

  it("calls onAcknowledge with correct parameters", () => {
    const onAcknowledge = jest.fn();
    render(<SignalCard signal={mockSignal} onAcknowledge={onAcknowledge} />);

    fireEvent.click(screen.getByText("Act"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "acted");

    fireEvent.click(screen.getByText("Seen"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "seen");

    fireEvent.click(screen.getByText("Skip"));
    expect(onAcknowledge).toHaveBeenCalledWith(1, "skipped");
  });

  it("shows green urgency for high confidence BUY", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} />);
    expect(screen.getByText("GREEN")).toBeInTheDocument();
  });

  it("shows red urgency for SELL action", () => {
    const sell = { ...mockSignal, action: "SELL" as const };
    render(<SignalCard signal={sell} onAcknowledge={() => {}} />);
    expect(screen.getByText("RED")).toBeInTheDocument();
  });

  it("disables buttons when loading", () => {
    render(<SignalCard signal={mockSignal} onAcknowledge={() => {}} isLoading />);

    const buttons = screen.getAllByRole("button");
    buttons.forEach((button) => {
      expect(button).toBeDisabled();
    });
  });
});
```

- [ ] **Step 7: Run tests**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm test -- __tests__/components/signal-card.test.tsx
```

Expected: All tests pass.

- [ ] **Step 8: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/components/charts/price-chart.tsx \
  services/nextjs/src/components/charts/performance-chart.tsx \
  services/nextjs/src/components/signals/signal-card.tsx \
  services/nextjs/src/components/portfolio/position-row.tsx \
  services/nextjs/__tests__/components/signal-card.test.tsx
git commit -m "feat: add chart components, signal card, and position row"
git push -u origin feat/task-4-chart-signal-components
gh pr create --title "feat: chart components + signal card + position row" --body "$(cat <<'EOF'
## Summary
- PriceChart: Recharts composed chart with OHLCV, volume bars
- PerformanceChart: area chart with gradient fill for equity curve
- SignalCard: urgency-coded signal display with one-click acknowledge (Act/Seen/Skip)
- PositionRow: portfolio table row with P&L, stop-loss proximity bar

## Test plan
- [ ] SignalCard renders all signal details
- [ ] Acknowledge buttons fire correct callbacks
- [ ] Urgency levels map correctly (green/yellow/red)
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 5: API Routes (BFF Layer)

**Files:**
- Create: `services/nextjs/src/app/api/signals/route.ts`
- Create: `services/nextjs/src/app/api/signals/[id]/route.ts`
- Create: `services/nextjs/src/app/api/portfolio/route.ts`
- Create: `services/nextjs/src/app/api/backtest/route.ts`
- Create: `services/nextjs/src/app/api/assets/[symbol]/route.ts`
- Create: `services/nextjs/src/app/api/audit/route.ts`
- Create: `services/nextjs/src/app/api/settings/route.ts`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-5-api-routes
```

- [ ] **Step 2: Create signals API route**

Create `services/nextjs/src/app/api/signals/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get("limit") || "50", 10);
    const offset = parseInt(searchParams.get("offset") || "0", 10);
    const asset = searchParams.get("asset");
    const action = searchParams.get("action");
    const acknowledged = searchParams.get("acknowledged");

    const where: Record<string, unknown> = {};
    if (asset) where.asset = asset.toUpperCase();
    if (action) where.action = action.toUpperCase();
    if (acknowledged !== null && acknowledged !== undefined && acknowledged !== "") {
      where.acknowledged = acknowledged === "true";
    }

    const [signals, total] = await Promise.all([
      prisma.signal.findMany({
        where,
        orderBy: { created_at: "desc" },
        take: limit,
        skip: offset,
      }),
      prisma.signal.count({ where }),
    ]);

    return NextResponse.json({
      data: signals.map((s) => ({
        ...s,
        created_at: s.created_at.toISOString(),
      })),
      total,
      limit,
      offset,
    });
  } catch (error) {
    console.error("GET /api/signals error:", error);
    return NextResponse.json(
      { error: "Failed to fetch signals" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 3: Create signal acknowledge API route**

Create `services/nextjs/src/app/api/signals/[id]/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function PATCH(
  request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id, 10);
    if (isNaN(id)) {
      return NextResponse.json({ error: "Invalid signal ID" }, { status: 400 });
    }

    const body = await request.json();
    const { acknowledged, acknowledge_status } = body;

    const signal = await prisma.signal.update({
      where: { id },
      data: { acknowledged: acknowledged ?? true },
    });

    // Log the acknowledgment to the audit log
    await prisma.auditLog.create({
      data: {
        event_type: "signal_acknowledged",
        asset: signal.asset,
        details: JSON.stringify({
          signal_id: id,
          action: signal.action,
          acknowledge_status: acknowledge_status || "seen",
          confidence: signal.confidence,
        }),
        created_at: new Date(),
      },
    });

    return NextResponse.json({
      data: {
        ...signal,
        created_at: signal.created_at.toISOString(),
      },
    });
  } catch (error) {
    console.error("PATCH /api/signals/[id] error:", error);
    return NextResponse.json(
      { error: "Failed to update signal" },
      { status: 500 }
    );
  }
}

export async function GET(
  _request: NextRequest,
  { params }: { params: { id: string } }
) {
  try {
    const id = parseInt(params.id, 10);
    if (isNaN(id)) {
      return NextResponse.json({ error: "Invalid signal ID" }, { status: 400 });
    }

    const signal = await prisma.signal.findUnique({ where: { id } });
    if (!signal) {
      return NextResponse.json({ error: "Signal not found" }, { status: 404 });
    }

    return NextResponse.json({
      data: {
        ...signal,
        created_at: signal.created_at.toISOString(),
      },
    });
  } catch (error) {
    console.error("GET /api/signals/[id] error:", error);
    return NextResponse.json(
      { error: "Failed to fetch signal" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 4: Create portfolio API route**

Create `services/nextjs/src/app/api/portfolio/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const status = searchParams.get("status");
    const asset = searchParams.get("asset");

    const where: Record<string, unknown> = {};
    if (status) where.status = status;
    if (asset) where.asset = asset.toUpperCase();

    const positions = await prisma.portfolio.findMany({
      where,
      orderBy: { opened_at: "desc" },
    });

    // For open positions, fetch the latest price from asset_prices_hourly
    const openPositions = positions.filter((p) => p.status === "open");
    const currentPrices: Record<string, number> = {};

    if (openPositions.length > 0) {
      const assets = [...new Set(openPositions.map((p) => p.asset))];
      for (const a of assets) {
        const latestPrice = await prisma.assetPriceHourly.findFirst({
          where: { asset: a },
          orderBy: { timestamp: "desc" },
          select: { close: true },
        });
        if (latestPrice) {
          currentPrices[a] = latestPrice.close;
        }
      }
    }

    const enrichedPositions = positions.map((p) => {
      const currentPrice = currentPrices[p.asset] ?? p.entry_price;
      const isOpen = p.status === "open";
      return {
        ...p,
        opened_at: p.opened_at.toISOString(),
        closed_at: p.closed_at?.toISOString() ?? null,
        current_price: isOpen ? currentPrice : undefined,
        unrealized_pnl_idr: isOpen
          ? Math.round(
              (currentPrice - p.entry_price) * p.quantity * 16000
            )
          : undefined,
        unrealized_pnl_pct: isOpen
          ? ((currentPrice - p.entry_price) / p.entry_price) * 100
          : undefined,
      };
    });

    // Compute portfolio summary
    const openCount = enrichedPositions.filter((p) => p.status === "open").length;
    const totalInvested = enrichedPositions
      .filter((p) => p.status === "open")
      .reduce((sum, p) => sum + p.entry_amount_idr, 0);
    const totalUnrealizedPnl = enrichedPositions
      .filter((p) => p.status === "open")
      .reduce((sum, p) => sum + (p.unrealized_pnl_idr ?? 0), 0);
    const totalRealizedPnl = enrichedPositions
      .filter((p) => p.status !== "open")
      .reduce((sum, p) => sum + (p.pnl_idr ?? 0), 0);

    return NextResponse.json({
      data: enrichedPositions,
      summary: {
        open_positions: openCount,
        total_invested_idr: totalInvested,
        total_unrealized_pnl_idr: totalUnrealizedPnl,
        total_realized_pnl_idr: totalRealizedPnl,
        portfolio_value_idr: totalInvested + totalUnrealizedPnl,
      },
    });
  } catch (error) {
    console.error("GET /api/portfolio error:", error);
    return NextResponse.json(
      { error: "Failed to fetch portfolio" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 5: Create backtest API route**

Create `services/nextjs/src/app/api/backtest/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(_request: NextRequest) {
  try {
    // Compute metrics from closed portfolio positions
    const closedPositions = await prisma.portfolio.findMany({
      where: { status: { in: ["closed", "stopped"] } },
      orderBy: { closed_at: "asc" },
    });

    if (closedPositions.length === 0) {
      return NextResponse.json({
        data: {
          metrics: {
            total_return_idr: 0,
            total_return_pct: 0,
            win_rate: 0,
            reward_risk_ratio: 0,
            sharpe_ratio: 0,
            max_drawdown_pct: 0,
            total_trades: 0,
            winning_trades: 0,
            losing_trades: 0,
          },
          monthly: [],
          equity_curve: [],
        },
      });
    }

    const totalTrades = closedPositions.length;
    const winningTrades = closedPositions.filter((p) => (p.pnl_idr ?? 0) > 0);
    const losingTrades = closedPositions.filter((p) => (p.pnl_idr ?? 0) <= 0);

    const totalReturnIdr = closedPositions.reduce(
      (sum, p) => sum + (p.pnl_idr ?? 0),
      0
    );
    const totalInvested = closedPositions.reduce(
      (sum, p) => sum + p.entry_amount_idr,
      0
    );
    const totalReturnPct =
      totalInvested > 0 ? (totalReturnIdr / totalInvested) * 100 : 0;
    const winRate =
      totalTrades > 0 ? (winningTrades.length / totalTrades) * 100 : 0;

    const avgWin =
      winningTrades.length > 0
        ? winningTrades.reduce((s, p) => s + (p.pnl_idr ?? 0), 0) /
          winningTrades.length
        : 0;
    const avgLoss =
      losingTrades.length > 0
        ? Math.abs(
            losingTrades.reduce((s, p) => s + (p.pnl_idr ?? 0), 0) /
              losingTrades.length
          )
        : 1;
    const rewardRiskRatio = avgLoss > 0 ? avgWin / avgLoss : 0;

    // Compute equity curve
    let cumulative = 0;
    const equityCurve = closedPositions.map((p) => {
      cumulative += p.pnl_idr ?? 0;
      return {
        date: (p.closed_at ?? p.opened_at).toISOString().split("T")[0],
        value: cumulative,
      };
    });

    // Compute max drawdown
    let peak = 0;
    let maxDrawdown = 0;
    for (const point of equityCurve) {
      if (point.value > peak) peak = point.value;
      const drawdown = peak > 0 ? ((peak - point.value) / peak) * 100 : 0;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }

    // Compute Sharpe ratio (simplified: using monthly returns)
    const returns = closedPositions.map(
      (p) => ((p.pnl_idr ?? 0) / p.entry_amount_idr) * 100
    );
    const avgReturn =
      returns.length > 0
        ? returns.reduce((s, r) => s + r, 0) / returns.length
        : 0;
    const stdDev =
      returns.length > 1
        ? Math.sqrt(
            returns.reduce((s, r) => s + Math.pow(r - avgReturn, 2), 0) /
              (returns.length - 1)
          )
        : 1;
    const sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(52) : 0;

    // Monthly breakdown
    const monthlyMap = new Map<
      string,
      { return_idr: number; trades: number; wins: number }
    >();
    for (const p of closedPositions) {
      const date = p.closed_at ?? p.opened_at;
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}`;
      const existing = monthlyMap.get(monthKey) || {
        return_idr: 0,
        trades: 0,
        wins: 0,
      };
      existing.return_idr += p.pnl_idr ?? 0;
      existing.trades += 1;
      if ((p.pnl_idr ?? 0) > 0) existing.wins += 1;
      monthlyMap.set(monthKey, existing);
    }

    const monthly = Array.from(monthlyMap.entries())
      .sort(([a], [b]) => a.localeCompare(b))
      .map(([month, data]) => ({
        month,
        return_idr: data.return_idr,
        return_pct:
          data.trades > 0 ? (data.return_idr / (data.trades * 1_000_000)) * 100 : 0,
        trades: data.trades,
        win_rate: data.trades > 0 ? (data.wins / data.trades) * 100 : 0,
      }));

    return NextResponse.json({
      data: {
        metrics: {
          total_return_idr: totalReturnIdr,
          total_return_pct: totalReturnPct,
          win_rate: winRate,
          reward_risk_ratio: rewardRiskRatio,
          sharpe_ratio: sharpeRatio,
          max_drawdown_pct: maxDrawdown,
          total_trades: totalTrades,
          winning_trades: winningTrades.length,
          losing_trades: losingTrades.length,
        },
        monthly,
        equity_curve: equityCurve,
      },
    });
  } catch (error) {
    console.error("GET /api/backtest error:", error);
    return NextResponse.json(
      { error: "Failed to fetch backtest results" },
      { status: 500 }
    );
  }
}

```

- [ ] **Step 6: Create asset detail API route**

Create `services/nextjs/src/app/api/assets/[symbol]/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(
  _request: NextRequest,
  { params }: { params: { symbol: string } }
) {
  try {
    const symbol = params.symbol.toUpperCase();

    // Fetch all data in parallel from PostgreSQL via Prisma
    const [priceHistory, fundamentals, recentSignals, tradeHistory, latestPrice] =
      await Promise.all([
        // Last 90 days of daily prices
        prisma.assetPriceDaily.findMany({
          where: { asset: symbol },
          orderBy: { date: "desc" },
          take: 90,
        }),
        // Latest fundamentals
        prisma.assetFundamental.findFirst({
          where: { asset: symbol },
          orderBy: { fetched_at: "desc" },
        }),
        // Last 20 signals
        prisma.signal.findMany({
          where: { asset: symbol },
          orderBy: { created_at: "desc" },
          take: 20,
        }),
        // All portfolio positions for this asset
        prisma.portfolio.findMany({
          where: { asset: symbol },
          orderBy: { opened_at: "desc" },
        }),
        // Latest price
        prisma.assetPriceHourly.findFirst({
          where: { asset: symbol },
          orderBy: { timestamp: "desc" },
          select: { close: true },
        }),
      ]);

    // Model scores and features are derived from the latest signal data
    // Python writes these to the database; we read them here
    const latestSignal = recentSignals[0];
    const modelScores = {
      xgboost: null as number | null,
      lightgbm: null as number | null,
      lstm: null as number | null,
      ensemble: latestSignal ? latestSignal.confidence : null,
    };
    const features: Record<string, number> = {};

    return NextResponse.json({
      data: {
        symbol,
        current_price: latestPrice?.close ?? 0,
        price_history: priceHistory
          .reverse()
          .map((p) => ({
            date: p.date.toISOString().split("T")[0],
            open: p.open,
            high: p.high,
            low: p.low,
            close: p.close,
            volume: p.volume,
          })),
        fundamentals: fundamentals
          ? {
              market_cap: fundamentals.market_cap,
              market_cap_rank: fundamentals.market_cap_rank,
              total_volume_24h: fundamentals.total_volume_24h,
              circulating_supply: fundamentals.circulating_supply,
              category: fundamentals.category,
            }
          : null,
        recent_signals: recentSignals.map((s) => ({
          ...s,
          created_at: s.created_at.toISOString(),
        })),
        trade_history: tradeHistory.map((p) => ({
          ...p,
          opened_at: p.opened_at.toISOString(),
          closed_at: p.closed_at?.toISOString() ?? null,
        })),
        model_scores: modelScores,
        features,
      },
    });
  } catch (error) {
    console.error("GET /api/assets/[symbol] error:", error);
    return NextResponse.json(
      { error: "Failed to fetch asset details" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 7: Create audit log API route**

Create `services/nextjs/src/app/api/audit/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    const limit = parseInt(searchParams.get("limit") || "100", 10);
    const offset = parseInt(searchParams.get("offset") || "0", 10);
    const eventType = searchParams.get("event_type");
    const asset = searchParams.get("asset");

    const where: Record<string, unknown> = {};
    if (eventType) where.event_type = eventType;
    if (asset) where.asset = asset.toUpperCase();

    const [entries, total] = await Promise.all([
      prisma.auditLog.findMany({
        where,
        orderBy: { created_at: "desc" },
        take: limit,
        skip: offset,
      }),
      prisma.auditLog.count({ where }),
    ]);

    return NextResponse.json({
      data: entries.map((e) => {
        let parsedDetails: Record<string, unknown> | undefined;
        if (e.details) {
          try {
            parsedDetails = JSON.parse(e.details);
          } catch {
            parsedDetails = undefined;
          }
        }
        return {
          ...e,
          created_at: e.created_at.toISOString(),
          parsed_details: parsedDetails,
        };
      }),
      total,
      limit,
      offset,
    });
  } catch (error) {
    console.error("GET /api/audit error:", error);
    return NextResponse.json(
      { error: "Failed to fetch audit log" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 8: Create settings API route**

Create `services/nextjs/src/app/api/settings/route.ts`:

```typescript
import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import type { RiskSettings } from "@/types";

const DEFAULT_SETTINGS: RiskSettings = {
  stop_loss_pct: -8,
  max_positions: 8,
  confidence_threshold: 0.7,
  position_size_idr: 1_000_000,
  max_single_asset_exposure_pct: 25,
  portfolio_drawdown_pause_pct: 20,
  retrain_schedule: "weekly",
};

export async function GET(_request: NextRequest) {
  try {
    // Read settings from audit_log where Python stores them,
    // or fall back to defaults
    const settingsEntry = await prisma.auditLog.findFirst({
      where: { event_type: "settings_updated" },
      orderBy: { created_at: "desc" },
    });

    if (settingsEntry?.details) {
      try {
        const settings = JSON.parse(settingsEntry.details) as RiskSettings;
        return NextResponse.json({ data: settings });
      } catch {
        // Fall back to defaults if parse fails
      }
    }

    return NextResponse.json({ data: DEFAULT_SETTINGS });
  } catch (error) {
    console.error("GET /api/settings error:", error);
    return NextResponse.json(
      { error: "Failed to fetch settings" },
      { status: 500 }
    );
  }
}

export async function PUT(request: NextRequest) {
  try {
    const body = await request.json();

    // Validate settings
    const settings: RiskSettings = {
      stop_loss_pct: parseFloat(body.stop_loss_pct),
      max_positions: parseInt(body.max_positions, 10),
      confidence_threshold: parseFloat(body.confidence_threshold),
      position_size_idr: parseInt(body.position_size_idr, 10),
      max_single_asset_exposure_pct: parseFloat(body.max_single_asset_exposure_pct),
      portfolio_drawdown_pause_pct: parseFloat(body.portfolio_drawdown_pause_pct),
      retrain_schedule: body.retrain_schedule,
    };

    // Validation rules
    if (settings.stop_loss_pct > 0 || settings.stop_loss_pct < -50) {
      return NextResponse.json(
        { error: "Stop loss must be between -50% and 0%" },
        { status: 400 }
      );
    }
    if (settings.max_positions < 1 || settings.max_positions > 20) {
      return NextResponse.json(
        { error: "Max positions must be between 1 and 20" },
        { status: 400 }
      );
    }
    if (settings.confidence_threshold < 0.1 || settings.confidence_threshold > 1.0) {
      return NextResponse.json(
        { error: "Confidence threshold must be between 0.1 and 1.0" },
        { status: 400 }
      );
    }
    if (settings.position_size_idr < 100_000 || settings.position_size_idr > 100_000_000) {
      return NextResponse.json(
        { error: "Position size must be between Rp 100.000 and Rp 100.000.000" },
        { status: 400 }
      );
    }

    // Save settings to the database via audit_log
    await prisma.auditLog.create({
      data: {
        event_type: "settings_updated",
        details: JSON.stringify(settings),
        created_at: new Date(),
      },
    });

    return NextResponse.json({ data: settings });
  } catch (error) {
    console.error("PUT /api/settings error:", error);
    return NextResponse.json(
      { error: "Failed to update settings" },
      { status: 500 }
    );
  }
}
```

- [ ] **Step 9: Verify type checking**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run type-check
```

Expected: No TypeScript errors.

- [ ] **Step 10: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/app/api/
git commit -m "feat: add API routes for signals, portfolio, backtest, assets, audit, settings"
git push -u origin feat/task-5-api-routes
gh pr create --title "feat: API routes for all dashboard pages" --body "$(cat <<'EOF'
## Summary
- GET/PATCH /api/signals - list signals with filtering, acknowledge with audit logging
- GET /api/portfolio - positions with real-time P&L via latest hourly price
- GET /api/backtest - compute metrics from closed portfolio positions in PostgreSQL
- GET /api/assets/[symbol] - full asset detail with prices, signals, model scores (all from PostgreSQL)
- GET /api/audit - paginated audit log with parsed JSON details
- GET/PUT /api/settings - risk parameter management with validation, persisted to database

## Test plan
- [ ] `npm run type-check` passes
- [ ] All routes return proper JSON responses
- [ ] Settings validation rejects invalid values
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 6: Overview Page + Signals Page

**Files:**
- Create: `services/nextjs/src/app/page.tsx`
- Create: `services/nextjs/src/app/signals/page.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-6-overview-signals-pages
```

- [ ] **Step 2: Create Overview page**

Create `services/nextjs/src/app/page.tsx`:

```typescript
import Link from "next/link";
import { prisma } from "@/lib/prisma";
import {
  formatIDR,
  formatIDRCompact,
  formatPercent,
  getPnlColor,
  getUrgencyLevel,
} from "@/lib/format";
import { PerformanceChart } from "@/components/charts/performance-chart";

async function getOverviewData() {
  const today = new Date();
  today.setHours(0, 0, 0, 0);

  const [openPositions, closedPositions, todaysSignals] = await Promise.all([
    prisma.portfolio.findMany({
      where: { status: "open" },
    }),
    prisma.portfolio.findMany({
      where: { status: { in: ["closed", "stopped"] } },
      orderBy: { closed_at: "asc" },
    }),
    prisma.signal.findMany({
      where: {
        created_at: { gte: today },
      },
      orderBy: { created_at: "desc" },
    }),
  ]);

  // Fetch current prices for open positions
  let totalInvested = 0;
  let totalUnrealizedPnl = 0;
  for (const pos of openPositions) {
    totalInvested += pos.entry_amount_idr;
    const latestPrice = await prisma.assetPriceHourly.findFirst({
      where: { asset: pos.asset },
      orderBy: { timestamp: "desc" },
      select: { close: true },
    });
    if (latestPrice) {
      const unrealized =
        (latestPrice.close - pos.entry_price) * pos.quantity * 16000;
      totalUnrealizedPnl += unrealized;
    }
  }

  const totalRealizedPnl = closedPositions.reduce(
    (sum, p) => sum + (p.pnl_idr ?? 0),
    0
  );

  const portfolioValue = totalInvested + totalUnrealizedPnl;
  const totalPnl = totalRealizedPnl + totalUnrealizedPnl;
  const totalPnlPct =
    totalInvested > 0 ? (totalPnl / totalInvested) * 100 : 0;

  // Build equity curve from closed positions
  let cumulative = 0;
  const equityCurve = closedPositions.map((p) => {
    cumulative += p.pnl_idr ?? 0;
    return {
      date: (p.closed_at ?? p.opened_at).toISOString().split("T")[0],
      value: cumulative,
    };
  });

  return {
    portfolioValue,
    totalPnl,
    totalPnlPct,
    openPositionsCount: openPositions.length,
    todaysSignals: todaysSignals.map((s) => ({
      ...s,
      created_at: s.created_at.toISOString(),
      urgency: getUrgencyLevel(s),
    })),
    equityCurve,
  };
}

export default async function OverviewPage() {
  const data = await getOverviewData();

  return (
    <div className="space-y-6">
      {/* Stats cards */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard
          title="Portfolio Value"
          value={formatIDR(data.portfolioValue)}
          subtitle="Total invested + unrealized P&L"
        />
        <StatCard
          title="Total P&L"
          value={formatIDR(data.totalPnl)}
          subtitle={formatPercent(data.totalPnlPct)}
          valueClassName={getPnlColor(data.totalPnl)}
        />
        <StatCard
          title="Open Positions"
          value={`${data.openPositionsCount} / 8`}
          subtitle="Active trades"
        />
        <StatCard
          title="Today's Signals"
          value={data.todaysSignals.length.toString()}
          subtitle={
            data.todaysSignals.length > 0
              ? `${data.todaysSignals.filter((s) => !s.acknowledged).length} unacknowledged`
              : "No new signals"
          }
        />
      </div>

      {/* Equity curve */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Portfolio Performance
        </h2>
        {data.equityCurve.length > 0 ? (
          <PerformanceChart data={data.equityCurve} startingValue={0} />
        ) : (
          <div className="flex h-64 items-center justify-center">
            <p className="text-sm text-slate-500">
              No closed trades yet. Performance chart will appear after your first trade closes.
            </p>
          </div>
        )}
      </div>

      {/* Today's signals */}
      <div className="card">
        <div className="mb-4 flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-100">
            Today&apos;s Signals
          </h2>
          <Link href="/signals" className="text-sm text-brand-400 hover:text-brand-300">
            View all signals
          </Link>
        </div>
        {data.todaysSignals.length === 0 ? (
          <p className="text-sm text-slate-500">
            No signals generated today. Check back after the daily signal generation runs.
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Asset</th>
                  <th className="table-header">Action</th>
                  <th className="table-header">Confidence</th>
                  <th className="table-header">Expected Return</th>
                  <th className="table-header">Hold</th>
                  <th className="table-header">Agreement</th>
                  <th className="table-header">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {data.todaysSignals.map((signal) => (
                  <tr
                    key={signal.id}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell">
                      <Link
                        href={`/assets/${signal.asset.toLowerCase()}`}
                        className="font-medium text-brand-400 hover:text-brand-300"
                      >
                        {signal.asset}
                      </Link>
                    </td>
                    <td className="table-cell">
                      <span
                        className={
                          signal.action === "BUY"
                            ? "text-green-400 font-semibold"
                            : signal.action === "SELL" || signal.action === "EXIT"
                              ? "text-red-400 font-semibold"
                              : "text-yellow-400 font-semibold"
                        }
                      >
                        {signal.action}
                      </span>
                    </td>
                    <td className="table-cell font-mono">
                      {Math.round(signal.confidence * 100)}%
                    </td>
                    <td className="table-cell">
                      <span className={getPnlColor(signal.expected_return_pct)}>
                        {formatPercent(signal.expected_return_pct)}
                      </span>
                    </td>
                    <td className="table-cell">{signal.suggested_hold_days}d</td>
                    <td className="table-cell">{signal.model_agreement}</td>
                    <td className="table-cell">
                      <span
                        className={
                          signal.urgency === "green"
                            ? "badge-green"
                            : signal.urgency === "yellow"
                              ? "badge-yellow"
                              : "badge-red"
                        }
                      >
                        {signal.urgency?.toUpperCase()}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  subtitle,
  valueClassName,
}: {
  title: string;
  value: string;
  subtitle: string;
  valueClassName?: string;
}) {
  return (
    <div className="card">
      <p className="text-sm text-slate-400">{title}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClassName || "text-slate-100"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
}
```

- [ ] **Step 3: Create Signals page**

Create `services/nextjs/src/app/signals/page.tsx`:

```typescript
"use client";

import { useEffect, useState, useCallback } from "react";
import { SignalCard } from "@/components/signals/signal-card";
import type { SignalRecord, AcknowledgeStatus } from "@/types";

export default function SignalsPage() {
  const [signals, setSignals] = useState<SignalRecord[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [acknowledging, setAcknowledging] = useState<number | null>(null);
  const [filter, setFilter] = useState<{
    action: string;
    acknowledged: string;
    asset: string;
  }>({
    action: "",
    acknowledged: "false",
    asset: "",
  });

  const fetchSignals = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filter.action) params.set("action", filter.action);
      if (filter.acknowledged) params.set("acknowledged", filter.acknowledged);
      if (filter.asset) params.set("asset", filter.asset);
      params.set("limit", "50");

      const response = await fetch(`/api/signals?${params.toString()}`);
      const result = await response.json();
      setSignals(result.data || []);
      setTotal(result.total || 0);
    } catch (error) {
      console.error("Failed to fetch signals:", error);
    } finally {
      setLoading(false);
    }
  }, [filter]);

  useEffect(() => {
    fetchSignals();
  }, [fetchSignals]);

  const handleAcknowledge = async (id: number, status: AcknowledgeStatus) => {
    setAcknowledging(id);
    try {
      const response = await fetch(`/api/signals/${id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ acknowledged: true, acknowledge_status: status }),
      });

      if (response.ok) {
        setSignals((prev) =>
          prev.map((s) => (s.id === id ? { ...s, acknowledged: true } : s))
        );
      }
    } catch (error) {
      console.error("Failed to acknowledge signal:", error);
    } finally {
      setAcknowledging(null);
    }
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Action</label>
            <select
              value={filter.action}
              onChange={(e) => setFilter({ ...filter, action: e.target.value })}
              className="input-field w-32"
            >
              <option value="">All</option>
              <option value="BUY">BUY</option>
              <option value="SELL">SELL</option>
              <option value="HOLD">HOLD</option>
              <option value="EXIT">EXIT</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Status</label>
            <select
              value={filter.acknowledged}
              onChange={(e) =>
                setFilter({ ...filter, acknowledged: e.target.value })
              }
              className="input-field w-40"
            >
              <option value="">All</option>
              <option value="false">Unacknowledged</option>
              <option value="true">Acknowledged</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Asset</label>
            <input
              type="text"
              value={filter.asset}
              onChange={(e) =>
                setFilter({ ...filter, asset: e.target.value.toUpperCase() })
              }
              placeholder="e.g. BTC"
              className="input-field w-28"
            />
          </div>
          <div className="flex items-end">
            <span className="text-sm text-slate-400">
              {total} signal{total !== 1 ? "s" : ""} found
            </span>
          </div>
        </div>
      </div>

      {/* Signal list */}
      {loading ? (
        <div className="space-y-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 w-24 rounded bg-slate-700" />
              <div className="mt-4 grid grid-cols-4 gap-4">
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
                <div className="h-8 rounded bg-slate-700" />
              </div>
            </div>
          ))}
        </div>
      ) : signals.length === 0 ? (
        <div className="card">
          <p className="text-center text-sm text-slate-500">
            No signals match your filters. Try adjusting the criteria above.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {signals.map((signal) => (
            <SignalCard
              key={signal.id}
              signal={signal}
              onAcknowledge={handleAcknowledge}
              isLoading={acknowledging === signal.id}
            />
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Verify build**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run type-check
npm run lint
```

Expected: No errors.

- [ ] **Step 5: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/app/page.tsx services/nextjs/src/app/signals/page.tsx
git commit -m "feat: add Overview and Signals pages with live data"
git push -u origin feat/task-6-overview-signals-pages
gh pr create --title "feat: Overview + Signals pages" --body "$(cat <<'EOF'
## Summary
- Overview page: portfolio value (IDR), total P&L, open positions count, today's signals table, equity curve chart
- Signals page: filterable signal list with action/status/asset filters, one-click acknowledge (Act/Seen/Skip), urgency badges, loading skeleton

## Test plan
- [ ] Overview shows stat cards and equity curve
- [ ] Signals page filters work (action, status, asset)
- [ ] Acknowledge buttons update signal state
- [ ] Loading skeleton displays during fetch
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 7: Portfolio Page + Backtest Page

**Files:**
- Create: `services/nextjs/src/app/portfolio/page.tsx`
- Create: `services/nextjs/src/app/backtest/page.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-7-portfolio-backtest-pages
```

- [ ] **Step 2: Create Portfolio page**

Create `services/nextjs/src/app/portfolio/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { PositionRow } from "@/components/portfolio/position-row";
import { formatIDR, formatPercent, getPnlColor } from "@/lib/format";
import type { PortfolioPosition } from "@/types";

interface PortfolioSummary {
  open_positions: number;
  total_invested_idr: number;
  total_unrealized_pnl_idr: number;
  total_realized_pnl_idr: number;
  portfolio_value_idr: number;
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PortfolioPosition[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState<string>("open");

  useEffect(() => {
    async function fetchPortfolio() {
      setLoading(true);
      try {
        const params = new URLSearchParams();
        if (statusFilter) params.set("status", statusFilter);
        const response = await fetch(`/api/portfolio?${params.toString()}`);
        const result = await response.json();
        setPositions(result.data || []);
        setSummary(result.summary || null);
      } catch (error) {
        console.error("Failed to fetch portfolio:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchPortfolio();
  }, [statusFilter]);

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      {summary && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div className="card">
            <p className="text-sm text-slate-400">Portfolio Value</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {formatIDR(summary.portfolio_value_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Open Positions</p>
            <p className="mt-1 text-2xl font-bold text-slate-100">
              {summary.open_positions} / 8
            </p>
            <p className="mt-1 text-xs text-slate-500">
              Invested: {formatIDR(summary.total_invested_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Unrealized P&L</p>
            <p
              className={`mt-1 text-2xl font-bold ${getPnlColor(summary.total_unrealized_pnl_idr)}`}
            >
              {formatIDR(summary.total_unrealized_pnl_idr)}
            </p>
          </div>
          <div className="card">
            <p className="text-sm text-slate-400">Realized P&L</p>
            <p
              className={`mt-1 text-2xl font-bold ${getPnlColor(summary.total_realized_pnl_idr)}`}
            >
              {formatIDR(summary.total_realized_pnl_idr)}
            </p>
          </div>
        </div>
      )}

      {/* Filter tabs */}
      <div className="flex gap-2">
        {["open", "closed", "stopped", ""].map((status) => (
          <button
            key={status || "all"}
            onClick={() => setStatusFilter(status)}
            className={`rounded-lg px-4 py-2 text-sm font-medium transition-colors ${
              statusFilter === status
                ? "bg-brand-600 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
            }`}
          >
            {status ? status.charAt(0).toUpperCase() + status.slice(1) : "All"}
          </button>
        ))}
      </div>

      {/* Positions table */}
      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="p-6">
            <div className="space-y-3">
              {[1, 2, 3].map((i) => (
                <div key={i} className="h-12 animate-pulse rounded bg-slate-700" />
              ))}
            </div>
          </div>
        ) : positions.length === 0 ? (
          <div className="p-6">
            <p className="text-center text-sm text-slate-500">
              No {statusFilter || ""} positions found.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Asset</th>
                  <th className="table-header">Status</th>
                  <th className="table-header">Entry Price</th>
                  <th className="table-header">Current Price</th>
                  <th className="table-header">Invested</th>
                  <th className="table-header">P&L (IDR)</th>
                  <th className="table-header">P&L (%)</th>
                  <th className="table-header">Stop Loss</th>
                  <th className="table-header">Opened</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <PositionRow key={position.id} position={position} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create Backtest page**

Create `services/nextjs/src/app/backtest/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { PerformanceChart } from "@/components/charts/performance-chart";
import {
  formatIDR,
  formatIDRCompact,
  formatPercent,
  getPnlColor,
} from "@/lib/format";
import type { BacktestResult } from "@/types";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from "recharts";

export default function BacktestPage() {
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchBacktest() {
      setLoading(true);
      try {
        const response = await fetch("/api/backtest");
        const data = await response.json();
        setResult(data.data || null);
      } catch (error) {
        console.error("Failed to fetch backtest:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchBacktest();
  }, []);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="card animate-pulse">
          <div className="h-8 w-48 rounded bg-slate-700" />
          <div className="mt-4 grid grid-cols-3 gap-4">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="h-16 rounded bg-slate-700" />
            ))}
          </div>
        </div>
        <div className="card animate-pulse">
          <div className="h-64 rounded bg-slate-700" />
        </div>
      </div>
    );
  }

  if (!result || result.metrics.total_trades === 0) {
    return (
      <div className="card">
        <p className="text-center text-sm text-slate-500">
          No backtest data available. Results will appear here once closed trades exist in the database.
        </p>
      </div>
    );
  }

  const { metrics, monthly, equity_curve } = result;

  return (
    <div className="space-y-6">
      {/* Metrics cards */}
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-3">
        <MetricCard
          title="Total Return"
          value={formatIDR(metrics.total_return_idr)}
          subtitle={formatPercent(metrics.total_return_pct)}
          valueClassName={getPnlColor(metrics.total_return_idr)}
        />
        <MetricCard
          title="Win Rate"
          value={formatPercent(metrics.win_rate, 1)}
          subtitle={`${metrics.winning_trades}W / ${metrics.losing_trades}L of ${metrics.total_trades} trades`}
        />
        <MetricCard
          title="Reward/Risk Ratio"
          value={metrics.reward_risk_ratio.toFixed(2)}
          subtitle="Average win / average loss"
        />
        <MetricCard
          title="Sharpe Ratio"
          value={metrics.sharpe_ratio.toFixed(2)}
          subtitle="Risk-adjusted return (annualized)"
        />
        <MetricCard
          title="Max Drawdown"
          value={formatPercent(-metrics.max_drawdown_pct, 1)}
          subtitle="Worst peak-to-trough drop"
          valueClassName="text-red-400"
        />
        <MetricCard
          title="Total Trades"
          value={metrics.total_trades.toString()}
          subtitle={`${metrics.winning_trades} winning, ${metrics.losing_trades} losing`}
        />
      </div>

      {/* Equity curve */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Equity Curve
        </h2>
        <PerformanceChart data={equity_curve} startingValue={0} />
      </div>

      {/* Monthly breakdown chart */}
      {monthly.length > 0 && (
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Monthly Returns
          </h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={monthly} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
              <XAxis
                dataKey="month"
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickLine={{ stroke: "#475569" }}
                axisLine={{ stroke: "#475569" }}
              />
              <YAxis
                tick={{ fill: "#94a3b8", fontSize: 12 }}
                tickLine={{ stroke: "#475569" }}
                axisLine={{ stroke: "#475569" }}
                tickFormatter={(value: number) => formatIDRCompact(value)}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1e293b",
                  border: "1px solid #334155",
                  borderRadius: "8px",
                  color: "#f1f5f9",
                }}
                formatter={(value: number, name: string) => {
                  if (name === "return_idr") return [formatIDR(value), "Return"];
                  return [value, name];
                }}
              />
              <Bar dataKey="return_idr" name="return_idr" radius={[4, 4, 0, 0]}>
                {monthly.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.return_idr >= 0 ? "#22c55e" : "#ef4444"}
                    opacity={0.8}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Monthly breakdown table */}
      {monthly.length > 0 && (
        <div className="card overflow-hidden p-0">
          <div className="px-6 pt-6">
            <h2 className="text-lg font-semibold text-slate-100">
              Monthly Breakdown
            </h2>
          </div>
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Month</th>
                  <th className="table-header">Return (IDR)</th>
                  <th className="table-header">Return (%)</th>
                  <th className="table-header">Trades</th>
                  <th className="table-header">Win Rate</th>
                </tr>
              </thead>
              <tbody>
                {monthly.map((m) => (
                  <tr
                    key={m.month}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell font-medium text-slate-200">
                      {m.month}
                    </td>
                    <td className={`table-cell font-mono ${getPnlColor(m.return_idr)}`}>
                      {formatIDR(m.return_idr)}
                    </td>
                    <td className={`table-cell font-mono ${getPnlColor(m.return_pct)}`}>
                      {formatPercent(m.return_pct)}
                    </td>
                    <td className="table-cell">{m.trades}</td>
                    <td className="table-cell">{formatPercent(m.win_rate, 1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function MetricCard({
  title,
  value,
  subtitle,
  valueClassName,
}: {
  title: string;
  value: string;
  subtitle: string;
  valueClassName?: string;
}) {
  return (
    <div className="card">
      <p className="text-sm text-slate-400">{title}</p>
      <p className={`mt-1 text-2xl font-bold ${valueClassName || "text-slate-100"}`}>
        {value}
      </p>
      <p className="mt-1 text-xs text-slate-500">{subtitle}</p>
    </div>
  );
}
```

- [ ] **Step 4: Verify build**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run type-check
npm run lint
```

Expected: No errors.

- [ ] **Step 5: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/app/portfolio/page.tsx services/nextjs/src/app/backtest/page.tsx
git commit -m "feat: add Portfolio and Backtest pages with metrics and charts"
git push -u origin feat/task-7-portfolio-backtest-pages
gh pr create --title "feat: Portfolio + Backtest pages" --body "$(cat <<'EOF'
## Summary
- Portfolio page: summary cards (value, positions, unrealized/realized P&L), status filter tabs, positions table with stop-loss proximity bar
- Backtest page: 6 metric cards (total return, win rate, reward/risk, Sharpe, max drawdown, total trades), equity curve chart, monthly returns bar chart with green/red coloring, monthly breakdown table

## Test plan
- [ ] Portfolio page shows summary and positions table
- [ ] Status filter tabs work (open/closed/stopped/all)
- [ ] Backtest metrics display correctly in IDR
- [ ] Monthly breakdown chart and table render
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 8: Asset Detail Page + Audit Log Page

**Files:**
- Create: `services/nextjs/src/app/assets/[symbol]/page.tsx`
- Create: `services/nextjs/src/app/audit/page.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-8-asset-audit-pages
```

- [ ] **Step 2: Create Asset Detail page**

Create `services/nextjs/src/app/assets/[symbol]/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { PriceChart } from "@/components/charts/price-chart";
import {
  formatUSD,
  formatIDR,
  formatPercent,
  formatConfidence,
  formatDateTime,
  getPnlColor,
  getUrgencyLevel,
  getUrgencyBadgeClass,
} from "@/lib/format";
import type { AssetDetail } from "@/types";

export default function AssetDetailPage() {
  const params = useParams();
  const symbol = (params.symbol as string)?.toUpperCase() || "";
  const [asset, setAsset] = useState<AssetDetail | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!symbol) return;

    async function fetchAsset() {
      setLoading(true);
      try {
        const response = await fetch(`/api/assets/${symbol.toLowerCase()}`);
        const result = await response.json();
        setAsset(result.data || null);
      } catch (error) {
        console.error("Failed to fetch asset:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchAsset();
  }, [symbol]);

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="card animate-pulse">
          <div className="h-8 w-32 rounded bg-slate-700" />
          <div className="mt-2 h-12 w-48 rounded bg-slate-700" />
        </div>
        <div className="card animate-pulse">
          <div className="h-96 rounded bg-slate-700" />
        </div>
      </div>
    );
  }

  if (!asset) {
    return (
      <div className="card">
        <p className="text-center text-sm text-slate-500">
          Asset &quot;{symbol}&quot; not found. No data available.
        </p>
        <div className="mt-4 text-center">
          <Link href="/" className="text-brand-400 hover:text-brand-300 text-sm">
            Back to Overview
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <h1 className="text-3xl font-bold text-slate-100">{asset.symbol}</h1>
          <p className="mt-1 text-2xl font-mono text-slate-200">
            {formatUSD(asset.current_price)}
          </p>
          {asset.fundamentals?.category && (
            <p className="mt-1 text-sm text-slate-500">
              {asset.fundamentals.category}
              {asset.fundamentals.market_cap_rank && (
                <span className="ml-2">
                  Rank #{asset.fundamentals.market_cap_rank}
                </span>
              )}
            </p>
          )}
        </div>
        <Link
          href="/portfolio"
          className="btn-secondary text-sm"
        >
          View Portfolio
        </Link>
      </div>

      {/* Price chart */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-slate-100">
          Price History (90 Days)
        </h2>
        <PriceChart data={asset.price_history} />
      </div>

      {/* Fundamentals + Model scores grid */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
        {/* Fundamentals */}
        {asset.fundamentals && (
          <div className="card">
            <h2 className="mb-4 text-lg font-semibold text-slate-100">
              Fundamentals
            </h2>
            <dl className="space-y-3">
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Market Cap</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.market_cap
                    ? `$${(asset.fundamentals.market_cap / 1e9).toFixed(1)}B`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Market Cap Rank</dt>
                <dd className="text-sm font-mono text-slate-200">
                  #{asset.fundamentals.market_cap_rank ?? "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">24h Volume</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.total_volume_24h
                    ? `$${(asset.fundamentals.total_volume_24h / 1e9).toFixed(2)}B`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Circulating Supply</dt>
                <dd className="text-sm font-mono text-slate-200">
                  {asset.fundamentals.circulating_supply
                    ? `${(asset.fundamentals.circulating_supply / 1e6).toFixed(2)}M`
                    : "N/A"}
                </dd>
              </div>
              <div className="flex justify-between">
                <dt className="text-sm text-slate-400">Category</dt>
                <dd className="text-sm text-slate-200">
                  {asset.fundamentals.category ?? "N/A"}
                </dd>
              </div>
            </dl>
          </div>
        )}

        {/* Model scores */}
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Model Scores
          </h2>
          <dl className="space-y-3">
            {(
              [
                ["XGBoost", asset.model_scores.xgboost],
                ["LightGBM", asset.model_scores.lightgbm],
                ["LSTM", asset.model_scores.lstm],
                ["Ensemble", asset.model_scores.ensemble],
              ] as [string, number | null][]
            ).map(([name, score]) => (
              <div key={name} className="flex items-center justify-between">
                <dt className="text-sm text-slate-400">{name}</dt>
                <dd className="flex items-center gap-3">
                  {score !== null ? (
                    <>
                      <div className="h-2 w-24 overflow-hidden rounded-full bg-slate-700">
                        <div
                          className="h-full rounded-full bg-brand-500"
                          style={{ width: `${Math.abs(score) * 100}%` }}
                        />
                      </div>
                      <span className="text-sm font-mono text-slate-200 w-12 text-right">
                        {formatConfidence(score)}
                      </span>
                    </>
                  ) : (
                    <span className="text-sm text-slate-500">No data</span>
                  )}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </div>

      {/* Feature values */}
      {Object.keys(asset.features).length > 0 && (
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-slate-100">
            Feature Values
          </h2>
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 sm:grid-cols-3 lg:grid-cols-4">
            {Object.entries(asset.features).map(([key, value]) => (
              <div key={key} className="flex justify-between border-b border-slate-800 py-1">
                <span className="text-xs text-slate-400 truncate mr-2">{key}</span>
                <span className="text-xs font-mono text-slate-300">
                  {typeof value === "number" ? value.toFixed(4) : String(value)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent signals */}
      <div className="card overflow-hidden p-0">
        <div className="px-6 pt-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Recent Signals
          </h2>
        </div>
        {asset.recent_signals.length === 0 ? (
          <div className="p-6">
            <p className="text-sm text-slate-500">No signals for this asset yet.</p>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Date</th>
                  <th className="table-header">Action</th>
                  <th className="table-header">Confidence</th>
                  <th className="table-header">Expected Return</th>
                  <th className="table-header">Hold</th>
                  <th className="table-header">Agreement</th>
                  <th className="table-header">Urgency</th>
                </tr>
              </thead>
              <tbody>
                {asset.recent_signals.map((signal) => {
                  const urgency = getUrgencyLevel(signal);
                  return (
                    <tr
                      key={signal.id}
                      className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                    >
                      <td className="table-cell text-xs">
                        {formatDateTime(signal.created_at)}
                      </td>
                      <td className="table-cell">
                        <span
                          className={
                            signal.action === "BUY"
                              ? "text-green-400 font-semibold"
                              : signal.action === "SELL" || signal.action === "EXIT"
                                ? "text-red-400 font-semibold"
                                : "text-yellow-400 font-semibold"
                          }
                        >
                          {signal.action}
                        </span>
                      </td>
                      <td className="table-cell font-mono">
                        {formatConfidence(signal.confidence)}
                      </td>
                      <td className={`table-cell font-mono ${getPnlColor(signal.expected_return_pct)}`}>
                        {formatPercent(signal.expected_return_pct)}
                      </td>
                      <td className="table-cell">{signal.suggested_hold_days}d</td>
                      <td className="table-cell">{signal.model_agreement}</td>
                      <td className="table-cell">
                        <span className={getUrgencyBadgeClass(urgency)}>
                          {urgency.toUpperCase()}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Trade history */}
      <div className="card overflow-hidden p-0">
        <div className="px-6 pt-6">
          <h2 className="text-lg font-semibold text-slate-100">
            Trade History
          </h2>
        </div>
        {asset.trade_history.length === 0 ? (
          <div className="p-6">
            <p className="text-sm text-slate-500">No trades for this asset yet.</p>
          </div>
        ) : (
          <div className="mt-4 overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header">Status</th>
                  <th className="table-header">Entry Price</th>
                  <th className="table-header">Exit Price</th>
                  <th className="table-header">Invested</th>
                  <th className="table-header">P&L (IDR)</th>
                  <th className="table-header">P&L (%)</th>
                  <th className="table-header">Opened</th>
                  <th className="table-header">Closed</th>
                </tr>
              </thead>
              <tbody>
                {asset.trade_history.map((trade) => (
                  <tr
                    key={trade.id}
                    className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50"
                  >
                    <td className="table-cell">
                      <span
                        className={`rounded-md px-2 py-0.5 text-xs font-semibold uppercase ${
                          trade.status === "open"
                            ? "bg-green-500/10 text-green-400"
                            : trade.status === "stopped"
                              ? "bg-red-500/10 text-red-400"
                              : "bg-slate-500/10 text-slate-400"
                        }`}
                      >
                        {trade.status}
                      </span>
                    </td>
                    <td className="table-cell font-mono">{formatUSD(trade.entry_price)}</td>
                    <td className="table-cell font-mono">
                      {trade.exit_price ? formatUSD(trade.exit_price) : "-"}
                    </td>
                    <td className="table-cell">{formatIDR(trade.entry_amount_idr)}</td>
                    <td className="table-cell">
                      {trade.pnl_idr !== null ? (
                        <span className={`font-mono ${getPnlColor(trade.pnl_idr)}`}>
                          {formatIDR(trade.pnl_idr)}
                        </span>
                      ) : (
                        <span className="text-slate-500">-</span>
                      )}
                    </td>
                    <td className="table-cell">
                      {trade.pnl_pct !== null ? (
                        <span className={`font-mono ${getPnlColor(trade.pnl_pct)}`}>
                          {formatPercent(trade.pnl_pct)}
                        </span>
                      ) : (
                        <span className="text-slate-500">-</span>
                      )}
                    </td>
                    <td className="table-cell text-xs">
                      {formatDateTime(trade.opened_at)}
                    </td>
                    <td className="table-cell text-xs">
                      {trade.closed_at ? formatDateTime(trade.closed_at) : "-"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Create Audit Log page**

Create `services/nextjs/src/app/audit/page.tsx`:

```typescript
"use client";

import { useEffect, useState, useCallback } from "react";
import { formatDateTime } from "@/lib/format";
import type { AuditEntry } from "@/types";

export default function AuditPage() {
  const [entries, setEntries] = useState<AuditEntry[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [offset, setOffset] = useState(0);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [filter, setFilter] = useState<{
    event_type: string;
    asset: string;
  }>({
    event_type: "",
    asset: "",
  });
  const limit = 50;

  const fetchAudit = useCallback(async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      params.set("limit", limit.toString());
      params.set("offset", offset.toString());
      if (filter.event_type) params.set("event_type", filter.event_type);
      if (filter.asset) params.set("asset", filter.asset);

      const response = await fetch(`/api/audit?${params.toString()}`);
      const result = await response.json();
      setEntries(result.data || []);
      setTotal(result.total || 0);
    } catch (error) {
      console.error("Failed to fetch audit log:", error);
    } finally {
      setLoading(false);
    }
  }, [offset, filter]);

  useEffect(() => {
    fetchAudit();
  }, [fetchAudit]);

  const totalPages = Math.ceil(total / limit);
  const currentPage = Math.floor(offset / limit) + 1;

  const eventTypeColors: Record<string, string> = {
    signal_generated: "text-blue-400 bg-blue-500/10",
    signal_acknowledged: "text-green-400 bg-green-500/10",
    position_opened: "text-emerald-400 bg-emerald-500/10",
    position_closed: "text-slate-400 bg-slate-500/10",
    stop_loss_triggered: "text-red-400 bg-red-500/10",
    model_retrained: "text-purple-400 bg-purple-500/10",
    data_fetched: "text-cyan-400 bg-cyan-500/10",
  };

  return (
    <div className="space-y-6">
      {/* Filters */}
      <div className="card">
        <div className="flex flex-wrap items-center gap-4">
          <div>
            <label className="block text-xs text-slate-400 mb-1">Event Type</label>
            <select
              value={filter.event_type}
              onChange={(e) => {
                setFilter({ ...filter, event_type: e.target.value });
                setOffset(0);
              }}
              className="input-field w-48"
            >
              <option value="">All Events</option>
              <option value="signal_generated">Signal Generated</option>
              <option value="signal_acknowledged">Signal Acknowledged</option>
              <option value="position_opened">Position Opened</option>
              <option value="position_closed">Position Closed</option>
              <option value="stop_loss_triggered">Stop Loss Triggered</option>
              <option value="model_retrained">Model Retrained</option>
              <option value="data_fetched">Data Fetched</option>
            </select>
          </div>
          <div>
            <label className="block text-xs text-slate-400 mb-1">Asset</label>
            <input
              type="text"
              value={filter.asset}
              onChange={(e) => {
                setFilter({ ...filter, asset: e.target.value.toUpperCase() });
                setOffset(0);
              }}
              placeholder="e.g. BTC"
              className="input-field w-28"
            />
          </div>
          <div className="flex items-end">
            <span className="text-sm text-slate-400">
              {total} event{total !== 1 ? "s" : ""} found
            </span>
          </div>
        </div>
      </div>

      {/* Audit log table */}
      <div className="card overflow-hidden p-0">
        {loading ? (
          <div className="p-6 space-y-3">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-10 animate-pulse rounded bg-slate-700" />
            ))}
          </div>
        ) : entries.length === 0 ? (
          <div className="p-6">
            <p className="text-center text-sm text-slate-500">
              No audit entries match your filters.
            </p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="table-header w-12">#</th>
                  <th className="table-header">Timestamp</th>
                  <th className="table-header">Event Type</th>
                  <th className="table-header">Asset</th>
                  <th className="table-header">Details</th>
                </tr>
              </thead>
              <tbody>
                {entries.map((entry) => {
                  const colorClass =
                    eventTypeColors[entry.event_type] ||
                    "text-slate-400 bg-slate-500/10";
                  const isExpanded = expandedId === entry.id;

                  return (
                    <tr
                      key={entry.id}
                      className="border-b border-[var(--border)] transition-colors hover:bg-slate-800/50 cursor-pointer"
                      onClick={() =>
                        setExpandedId(isExpanded ? null : entry.id)
                      }
                    >
                      <td className="table-cell text-xs text-slate-500 font-mono">
                        {entry.id}
                      </td>
                      <td className="table-cell text-xs">
                        {formatDateTime(entry.created_at)}
                      </td>
                      <td className="table-cell">
                        <span
                          className={`rounded-md px-2 py-0.5 text-xs font-medium ${colorClass}`}
                        >
                          {entry.event_type.replace(/_/g, " ")}
                        </span>
                      </td>
                      <td className="table-cell font-medium text-slate-200">
                        {entry.asset || "-"}
                      </td>
                      <td className="table-cell">
                        {isExpanded && entry.parsed_details ? (
                          <pre className="mt-1 max-w-md overflow-x-auto rounded bg-slate-800 p-2 text-xs text-slate-300 font-mono">
                            {JSON.stringify(entry.parsed_details, null, 2)}
                          </pre>
                        ) : entry.parsed_details ? (
                          <span className="text-xs text-slate-400 truncate block max-w-xs">
                            {JSON.stringify(entry.parsed_details).substring(0, 80)}
                            {JSON.stringify(entry.parsed_details).length > 80 ? "..." : ""}
                          </span>
                        ) : (
                          <span className="text-xs text-slate-500">
                            {entry.details
                              ? entry.details.substring(0, 80) +
                                (entry.details.length > 80 ? "..." : "")
                              : "-"}
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-[var(--border)] px-6 py-3">
            <p className="text-sm text-slate-400">
              Page {currentPage} of {totalPages}
            </p>
            <div className="flex gap-2">
              <button
                onClick={() => setOffset(Math.max(0, offset - limit))}
                disabled={offset === 0}
                className="btn-secondary text-xs"
              >
                Previous
              </button>
              <button
                onClick={() => setOffset(offset + limit)}
                disabled={offset + limit >= total}
                className="btn-secondary text-xs"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Verify build**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run type-check
npm run lint
```

Expected: No errors.

- [ ] **Step 5: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/app/assets/ services/nextjs/src/app/audit/
git commit -m "feat: add Asset Detail and Audit Log pages"
git push -u origin feat/task-8-asset-audit-pages
gh pr create --title "feat: Asset Detail + Audit Log pages" --body "$(cat <<'EOF'
## Summary
- Asset Detail page: price chart (90d), fundamentals card, model scores with progress bars, feature values grid, recent signals table, trade history table
- Audit Log page: filterable event log with event type/asset filters, expandable JSON details, pagination, color-coded event types

## Test plan
- [ ] Asset detail loads price chart and fundamentals
- [ ] Model scores display with progress bars
- [ ] Audit log pagination works
- [ ] Expandable details show formatted JSON
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 9: Settings Page

**Files:**
- Create: `services/nextjs/src/app/settings/page.tsx`

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-9-settings-page
```

- [ ] **Step 2: Create Settings page**

Create `services/nextjs/src/app/settings/page.tsx`:

```typescript
"use client";

import { useEffect, useState } from "react";
import { formatIDR } from "@/lib/format";
import type { RiskSettings } from "@/types";

export default function SettingsPage() {
  const [settings, setSettings] = useState<RiskSettings | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState<{
    type: "success" | "error";
    text: string;
  } | null>(null);
  const [formValues, setFormValues] = useState<RiskSettings>({
    stop_loss_pct: -8,
    max_positions: 8,
    confidence_threshold: 0.7,
    position_size_idr: 1_000_000,
    max_single_asset_exposure_pct: 25,
    portfolio_drawdown_pause_pct: 20,
    retrain_schedule: "weekly",
  });

  useEffect(() => {
    async function fetchSettings() {
      setLoading(true);
      try {
        const response = await fetch("/api/settings");
        const result = await response.json();
        if (result.data) {
          setSettings(result.data);
          setFormValues(result.data);
        }
      } catch (error) {
        console.error("Failed to fetch settings:", error);
      } finally {
        setLoading(false);
      }
    }

    fetchSettings();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setSaving(true);
    setSaveMessage(null);

    try {
      const response = await fetch("/api/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(formValues),
      });

      const result = await response.json();

      if (response.ok) {
        setSettings(result.data);
        setSaveMessage({ type: "success", text: "Settings saved successfully." });
      } else {
        setSaveMessage({
          type: "error",
          text: result.error || "Failed to save settings.",
        });
      }
    } catch (error) {
      console.error("Failed to save settings:", error);
      setSaveMessage({ type: "error", text: "Network error. Please try again." });
    } finally {
      setSaving(false);
      setTimeout(() => setSaveMessage(null), 5000);
    }
  };

  const handleChange = (field: keyof RiskSettings, value: string | number) => {
    setFormValues((prev) => ({ ...prev, [field]: value }));
  };

  const hasChanges =
    settings !== null &&
    JSON.stringify(settings) !== JSON.stringify(formValues);

  if (loading) {
    return (
      <div className="max-w-2xl space-y-6">
        <div className="card animate-pulse">
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="h-12 rounded bg-slate-700" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-2xl space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Risk Management */}
        <div className="card">
          <h2 className="mb-6 text-lg font-semibold text-slate-100">
            Risk Management
          </h2>
          <div className="space-y-5">
            {/* Stop Loss */}
            <SettingsField
              label="Stop Loss (%)"
              description="Hard stop-loss percentage from entry price. Must be negative."
              hint="Current: stop at {value}% loss"
              hintValue={formValues.stop_loss_pct.toString()}
            >
              <input
                type="number"
                value={formValues.stop_loss_pct}
                onChange={(e) =>
                  handleChange("stop_loss_pct", parseFloat(e.target.value))
                }
                min={-50}
                max={0}
                step={0.5}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Max Positions */}
            <SettingsField
              label="Max Concurrent Positions"
              description="Maximum number of open positions at any time."
              hint={`Max exposure: ${formatIDR(formValues.max_positions * formValues.position_size_idr)}`}
            >
              <input
                type="number"
                value={formValues.max_positions}
                onChange={(e) =>
                  handleChange("max_positions", parseInt(e.target.value, 10))
                }
                min={1}
                max={20}
                step={1}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Confidence Threshold */}
            <SettingsField
              label="Confidence Threshold"
              description="Minimum ensemble confidence to generate a signal. Range: 0.1 to 1.0."
              hint={`Signals require ${Math.round(formValues.confidence_threshold * 100)}%+ confidence`}
            >
              <input
                type="number"
                value={formValues.confidence_threshold}
                onChange={(e) =>
                  handleChange(
                    "confidence_threshold",
                    parseFloat(e.target.value)
                  )
                }
                min={0.1}
                max={1.0}
                step={0.05}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Position Size */}
            <SettingsField
              label="Position Size (IDR)"
              description="Fixed amount in IDR per trade entry."
              hint={`Current: ${formatIDR(formValues.position_size_idr)} per trade`}
            >
              <input
                type="number"
                value={formValues.position_size_idr}
                onChange={(e) =>
                  handleChange(
                    "position_size_idr",
                    parseInt(e.target.value, 10)
                  )
                }
                min={100_000}
                max={100_000_000}
                step={100_000}
                className="input-field w-48"
              />
            </SettingsField>

            {/* Max Single Asset Exposure */}
            <SettingsField
              label="Max Single Asset Exposure (%)"
              description="Maximum percentage of total capital in a single asset."
            >
              <input
                type="number"
                value={formValues.max_single_asset_exposure_pct}
                onChange={(e) =>
                  handleChange(
                    "max_single_asset_exposure_pct",
                    parseFloat(e.target.value)
                  )
                }
                min={5}
                max={100}
                step={5}
                className="input-field w-32"
              />
            </SettingsField>

            {/* Portfolio Drawdown Pause */}
            <SettingsField
              label="Portfolio Drawdown Pause (%)"
              description="Pause new signals if total portfolio drops this much from peak."
            >
              <input
                type="number"
                value={formValues.portfolio_drawdown_pause_pct}
                onChange={(e) =>
                  handleChange(
                    "portfolio_drawdown_pause_pct",
                    parseFloat(e.target.value)
                  )
                }
                min={5}
                max={50}
                step={5}
                className="input-field w-32"
              />
            </SettingsField>
          </div>
        </div>

        {/* Schedule */}
        <div className="card">
          <h2 className="mb-6 text-lg font-semibold text-slate-100">
            Model & Data Schedule
          </h2>
          <div className="space-y-5">
            <SettingsField
              label="Retrain Schedule"
              description="How often to retrain ML models with latest data."
            >
              <select
                value={formValues.retrain_schedule}
                onChange={(e) =>
                  handleChange("retrain_schedule", e.target.value)
                }
                className="input-field w-40"
              >
                <option value="daily">Daily</option>
                <option value="weekly">Weekly (Sunday)</option>
                <option value="biweekly">Bi-weekly</option>
                <option value="monthly">Monthly</option>
              </select>
            </SettingsField>

            <div className="rounded-lg border border-slate-700 bg-slate-800/50 p-4">
              <h3 className="text-sm font-medium text-slate-300">
                Fixed Schedules
              </h3>
              <ul className="mt-2 space-y-1 text-xs text-slate-400">
                <li>
                  <span className="text-slate-300">Hourly:</span> Fetch latest candle for all 20 assets, run ETL
                </li>
                <li>
                  <span className="text-slate-300">Daily (00:00 UTC):</span> Aggregate hourly to daily, generate signals, re-evaluate positions
                </li>
                <li>
                  <span className="text-slate-300">Weekly (Sunday):</span> Refresh fundamental data, update asset universe
                </li>
              </ul>
            </div>
          </div>
        </div>

        {/* Save button */}
        <div className="flex items-center gap-4">
          <button
            type="submit"
            disabled={saving || !hasChanges}
            className="btn-primary"
          >
            {saving ? "Saving..." : "Save Settings"}
          </button>
          {hasChanges && (
            <button
              type="button"
              onClick={() => settings && setFormValues(settings)}
              className="btn-secondary"
            >
              Reset
            </button>
          )}
          {saveMessage && (
            <span
              className={`text-sm ${
                saveMessage.type === "success"
                  ? "text-green-400"
                  : "text-red-400"
              }`}
            >
              {saveMessage.text}
            </span>
          )}
        </div>
      </form>
    </div>
  );
}

function SettingsField({
  label,
  description,
  hint,
  hintValue,
  children,
}: {
  label: string;
  description: string;
  hint?: string;
  hintValue?: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
      <div className="flex-1">
        <label className="block text-sm font-medium text-slate-200">
          {label}
        </label>
        <p className="mt-0.5 text-xs text-slate-500">{description}</p>
        {hint && (
          <p className="mt-0.5 text-xs text-slate-400">
            {hintValue ? hint.replace("{value}", hintValue) : hint}
          </p>
        )}
      </div>
      <div className="sm:ml-4">{children}</div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run type-check
npm run lint
```

Expected: No errors.

- [ ] **Step 4: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add services/nextjs/src/app/settings/page.tsx
git commit -m "feat: add Settings page with risk parameter management"
git push -u origin feat/task-9-settings-page
gh pr create --title "feat: Settings page with risk parameters" --body "$(cat <<'EOF'
## Summary
- Settings page with full risk parameter management form
- Fields: stop-loss %, max positions, confidence threshold, position size (IDR), max single asset exposure, portfolio drawdown pause, retrain schedule
- Form validation with min/max constraints
- Save/Reset buttons with success/error feedback
- Fixed schedule reference (hourly, daily, weekly)

## Test plan
- [ ] All form fields render with current values
- [ ] Validation prevents invalid inputs
- [ ] Save button disabled when no changes
- [ ] Reset button restores original values
- [ ] Success/error messages display after save
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```

---

### Task 10: GitHub Actions CI + Final Integration

**Files:**
- Create: `.github/workflows/nextjs-ci.yml`
- Modify: `services/nextjs/package.json` (verify scripts)

- [ ] **Step 1: Create a new branch**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git checkout -b feat/task-10-nextjs-ci
```

- [ ] **Step 2: Create GitHub Actions CI workflow**

Create `.github/workflows/nextjs-ci.yml`:

```yaml
name: Next.js CI

on:
  push:
    branches: [main]
    paths:
      - "services/nextjs/**"
      - ".github/workflows/nextjs-ci.yml"
  pull_request:
    branches: [main]
    paths:
      - "services/nextjs/**"
      - ".github/workflows/nextjs-ci.yml"

jobs:
  lint-typecheck:
    name: Lint & Type Check
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/nextjs

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: services/nextjs/package-lock.json

      - name: Install dependencies
        run: npm ci 2>/dev/null || npm install

      - name: Generate Prisma client
        run: npx prisma generate

      - name: ESLint
        run: npm run lint

      - name: TypeScript type check
        run: npm run type-check

  test:
    name: Tests
    runs-on: ubuntu-latest
    defaults:
      run:
        working-directory: services/nextjs

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: services/nextjs/package-lock.json

      - name: Install dependencies
        run: npm ci 2>/dev/null || npm install

      - name: Generate Prisma client
        run: npx prisma generate

      - name: Run tests
        run: npm run test:ci

  build:
    name: Build
    runs-on: ubuntu-latest
    needs: [lint-typecheck, test]
    defaults:
      run:
        working-directory: services/nextjs

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: "20"
          cache: "npm"
          cache-dependency-path: services/nextjs/package-lock.json

      - name: Install dependencies
        run: npm ci 2>/dev/null || npm install

      - name: Generate Prisma client
        run: npx prisma generate

      - name: Build
        run: npm run build
        env:
          DATABASE_URL: "postgresql://dummy:dummy@localhost:5432/dummy"

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: [build]

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Build Docker image
        run: docker build -t ai-finance-nextjs ./services/nextjs
```

- [ ] **Step 3: Verify all tests pass locally**

```bash
cd /c/Users/togat/Desktop/AI-Finance/services/nextjs
npm run lint
npm run type-check
npm test -- --passWithNoTests
```

Expected: All checks pass.

- [ ] **Step 4: Verify Docker build**

```bash
cd /c/Users/togat/Desktop/AI-Finance
docker build -t ai-finance-nextjs ./services/nextjs
```

Expected: Docker image builds successfully.

- [ ] **Step 5: Commit and push**

```bash
cd /c/Users/togat/Desktop/AI-Finance
git add .github/workflows/nextjs-ci.yml
git commit -m "ci: add GitHub Actions workflow for Next.js lint, test, build, Docker"
git push -u origin feat/task-10-nextjs-ci
gh pr create --title "ci: Next.js CI workflow" --body "$(cat <<'EOF'
## Summary
- GitHub Actions CI for Next.js dashboard
- Jobs: ESLint + type-check, Jest tests with coverage, production build, Docker image build
- Triggers on push/PR to main when services/nextjs/** changes
- Uses Node.js 20, npm caching, Prisma client generation

## Test plan
- [ ] Workflow triggers on PR
- [ ] Lint + type-check passes
- [ ] Tests pass
- [ ] Build succeeds
- [ ] Docker build succeeds
EOF
)"
```

After PR is merged:
```bash
git checkout main && git pull
```
