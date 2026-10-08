import { cookies } from "next/headers";
import { getRequestConfig } from "next-intl/server";

// Use fixed import paths so Next.js can include both catalogs in the build.
import enMessages from "../../messages/en.json";
import zhCNMessages from "../../messages/zh-CN.json";

// Match each supported language to the messages shown by the interface.
const catalogs = {
  en: enMessages,
  "zh-CN": zhCNMessages,
} as const;

// Derive the allowed language names from the catalog map above.
type AppLocale = keyof typeof catalogs;

export default getRequestConfig(async () => {
  // Read the language saved by the English / 中文 selector.
  const cookieStore = await cookies();
  const savedLocale = cookieStore.get("locale")?.value;

  // Use Chinese only for the supported zh-CN value; otherwise default to English.
  const locale: AppLocale = savedLocale === "zh-CN" ? "zh-CN" : "en";

  // Return both the active language and its matching message catalog.
  return {
    locale,
    messages: catalogs[locale],
  };
});
