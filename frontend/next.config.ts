import createNextIntlPlugin from "next-intl/plugin";

// Keep any existing Next.js settings inside this object.
const nextConfig = {
  // Existing settings stay here.
};

const withNextIntl = createNextIntlPlugin();

export default withNextIntl(nextConfig);