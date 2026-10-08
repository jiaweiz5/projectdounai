import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { NextIntlClientProvider } from "next-intl";
import { getLocale } from "next-intl/server";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

// Set the browser tab title and description in the selected language.
export async function generateMetadata(): Promise<Metadata> {
  const locale = await getLocale();

  if (locale === "zh-CN") {
    return {
      title: "小红书内容核验",
      description: "分析小红书帖子正文、评论和配图，辅助评估内容可信度。",
    };
  }

  return {
    title: "XHS Content Verifier",
    description:
      "Analyze Xiaohongshu post text, comments, and images to assess content authenticity.",
  };
}

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Read the language selected by the user.
  const locale = await getLocale();

  return (
    <html
      lang={locale}
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col">
        {/* Makes the selected language messages available to the page and components. */}
        <NextIntlClientProvider>{children}</NextIntlClientProvider>
      </body>
    </html>
  );
}