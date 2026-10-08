"use client";

import { useRouter } from "next/navigation";

type Locale = "en" | "zh-CN";

export default function LanguageSwitcher() {
  const router = useRouter();

  function chooseLanguage(locale: Locale) {
    // Save the choice so the server loads the matching message file.
    document.cookie =
      `locale=${locale}; Path=/; Max-Age=31536000; SameSite=Lax`;

    // Refresh the page's translated messages without changing its URL.
    router.refresh();
  }

  return (
    <div className="flex gap-2" aria-label="Language">
      <button type="button" onClick={() => chooseLanguage("en")}>
        English
      </button>
      <button type="button" onClick={() => chooseLanguage("zh-CN")}>
        中文
      </button>
    </div>
  );
}