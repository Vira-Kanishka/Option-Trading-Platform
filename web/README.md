# Web app

`amoghopaya_app.jsx` is the React prototype — a single-file component implementing the full Amoghopāya UI.

## Running it

The fastest way to see it is the live demo linked in the root README.

To run locally you need a Vite + React + Tailwind scaffold:

```bash
npm create vite@latest amoghopaya -- --template react
cd amoghopaya
npm install recharts lucide-react
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
```

Then:

1. Replace `src/App.jsx` with the contents of `amoghopaya_app.jsx` (and update the import in `src/main.jsx` accordingly).
2. Add the Tailwind directives to `src/index.css`:
   ```css
   @tailwind base;
   @tailwind components;
   @tailwind utilities;
   ```
3. Point `tailwind.config.js` `content` at `./src/**/*.{js,jsx}` and safelist the arbitrary colour values used in dynamic class strings (e.g. `bg-[#2E4A7B]`).
4. `npm run dev`.

## Dependencies

- React 18
- Recharts 2.x (charts)
- lucide-react (icons)
- Tailwind CSS 3.x (styling)

## Relationship to the Python engine

The web app contains a JavaScript reimplementation of the pricing maths so it can run entirely in the browser with no backend. The authoritative reference implementation is the Python `engine/`. The two are kept numerically consistent; the Python suite is the source of truth for correctness.
