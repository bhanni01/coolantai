import { PROFILE_FIELDS } from '../lib/profile'
import type { DatacenterProfile } from '../lib/profile'

interface ProfileFormProps {
  profile: DatacenterProfile
  onChange: (profile: DatacenterProfile) => void
  disabled: boolean
}

/**
 * The only run input: five closed dropdowns mirroring the backend's
 * DatacenterProfile Literals. The full TargetSpec is resolved server-side, so
 * nothing here is free text.
 */
export function ProfileForm({ profile, onChange, disabled }: ProfileFormProps) {
  return (
    <div className="grid grid-cols-1 gap-3 p-4 sm:grid-cols-2 lg:grid-cols-5">
      {PROFILE_FIELDS.map((field) => (
        <label key={field.key} className="flex min-w-0 flex-col gap-1.5">
          <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-ink-faint">
            {field.label}
          </span>
          <select
            value={profile[field.key]}
            disabled={disabled}
            onChange={(e) =>
              onChange({ ...profile, [field.key]: e.target.value } as DatacenterProfile)
            }
            className="w-full appearance-none rounded-lg border border-hairline bg-surface-raised px-2.5 py-2 font-sans text-sm text-ink transition-colors hover:border-ink-faint focus:border-teal/60 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
          >
            {field.options.map((o) => (
              <option key={o.value} value={o.value}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
      ))}
    </div>
  )
}
