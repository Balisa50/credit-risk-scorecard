import { Dashboard } from "@/components/Dashboard";
import type { PipelineData } from "@/lib/types";
import data from "@/public/data/pipeline_results.json";

export default function Home() {
  return <Dashboard data={data as unknown as PipelineData} />;
}
