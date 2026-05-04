-- CreateTable
CREATE TABLE "v_new_1_shadow_signals" (
    "id"           SERIAL       NOT NULL,
    "timestamp"    TIMESTAMPTZ  NOT NULL,
    "symbol"       VARCHAR(20)  NOT NULL,
    "direction"    VARCHAR(10)  NOT NULL,
    "score_raw"    DOUBLE PRECISION NOT NULL,
    "score_platt"  DOUBLE PRECISION NOT NULL,
    "rank"         INTEGER      NOT NULL,
    "cfgi_value"   INTEGER      NOT NULL,
    "action"       VARCHAR(40)  NOT NULL,
    "market_close" DOUBLE PRECISION NOT NULL,
    "created_at"   TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "v_new_1_shadow_signals_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "v_new_1_shadow_signals_timestamp_idx" ON "v_new_1_shadow_signals"("timestamp");

-- CreateIndex
CREATE INDEX "v_new_1_shadow_signals_symbol_timestamp_idx" ON "v_new_1_shadow_signals"("symbol", "timestamp");

-- CreateIndex
CREATE INDEX "v_new_1_shadow_signals_direction_action_idx" ON "v_new_1_shadow_signals"("direction", "action");

-- UniqueConstraint (required for ON CONFLICT upsert in shadow scorer)
ALTER TABLE "v_new_1_shadow_signals"
  ADD CONSTRAINT "v_new_1_shadow_signals_timestamp_symbol_direction_key"
  UNIQUE ("timestamp", "symbol", "direction");
