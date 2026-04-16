import bcrypt from "bcryptjs";
import NextAuth from "next-auth";
import Credentials from "next-auth/providers/credentials";
import { db } from "@/lib/db";

const LOCAL_MODE_EMAIL = "local@agentic.app";
const LOCAL_MODE_NAME = "Local Workspace";
const LOCAL_MODE_PASSWORD = "local-mode-only";
const LOCAL_MODE_USER = {
  id: "local-workspace",
  email: LOCAL_MODE_EMAIL,
  name: LOCAL_MODE_NAME
};

export const { auth, handlers, signIn, signOut } = NextAuth({
  pages: {
    signIn: "/login"
  },
  session: {
    strategy: "jwt"
  },
  providers: [
    Credentials({
      name: "Credentials",
      credentials: {
        email: { label: "Email", type: "email" },
        password: { label: "Password", type: "password" }
      },
      async authorize(credentials) {
        if (!credentials?.email || !credentials.password) {
          return null;
        }

        const user = await db.user.findUnique({
          where: {
            email: credentials.email as string
          }
        });

        if (!user) {
          return null;
        }

        const isValid = await bcrypt.compare(credentials.password as string, user.passwordHash);
        if (!isValid) {
          return null;
        }

        return {
          id: user.id,
          email: user.email,
          name: user.name
        };
      }
    })
  ],
  callbacks: {
    jwt({ token, user }) {
      if (user) {
        token.id = user.id;
      }

      return token;
    },
    session({ session, token }) {
      if (session.user) {
        session.user.id = token.id as string;
      }

      return session;
    }
  }
});

function isDynamicServerUsageError(error: unknown) {
  if (!error || typeof error !== "object") {
    return false;
  }

  const candidate = error as { digest?: string; description?: string; message?: string };
  return (
    candidate.digest === "DYNAMIC_SERVER_USAGE" ||
    candidate.description?.includes("Dynamic server usage") === true ||
    candidate.message?.includes("Dynamic server usage") === true
  );
}

export async function getCurrentUser() {
  try {
    const session = await auth();
    if (session?.user) {
      return session.user;
    }
  } catch (error) {
    if (!isDynamicServerUsageError(error) && process.env.NODE_ENV !== "production") {
      console.warn("Auth session lookup failed, falling back to local mode.", error);
    }
  }

  return LOCAL_MODE_USER;
}

export async function requireUser() {
  return getCurrentUser();
}
