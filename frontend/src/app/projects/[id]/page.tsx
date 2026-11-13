import ProjectClient from "./ProjectClient";

export function generateStaticParams() {
  return [
    { id: "realistic-room" },
    { id: "demo-room" },
  ];
}

export default async function ProjectPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <ProjectClient projectId={id} />;
}
