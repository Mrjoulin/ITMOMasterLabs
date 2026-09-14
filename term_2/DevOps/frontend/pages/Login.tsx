import React, { useState } from 'react';
import { api } from '../services/api';

interface LoginProps {
  onLogin: () => void;
  onSwitchToSignUp: () => void;
}

const Login: React.FC<LoginProps> = ({ onLogin, onSwitchToSignUp }) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append('username', email);
      formData.append('password', password);
      await api.login(formData);
      onLogin();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-4 relative overflow-hidden bg-background-light dark:bg-background-dark">
      {/* Background decoration */}
      <div className="absolute inset-0 z-0 opacity-40 pointer-events-none">
        <div className="absolute -top-[10%] -left-[10%] w-[50%] h-[50%] rounded-full bg-primary/10 blur-[120px]"></div>
        <div className="absolute -bottom-[10%] left-[20%] w-[40%] h-[40%] rounded-full bg-primary/10 blur-[100px]"></div>
      </div>

      <div className="relative z-10 w-full max-w-md p-8 bg-white dark:bg-slate-900 shadow-xl rounded-xl border border-slate-100 dark:border-slate-800 animate-fade-in-up">
        <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-primary/10 text-primary mb-4">
                <span className="material-icons text-3xl">check_circle</span>
            </div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900 dark:text-white">Welcome back</h2>
            <p className="mt-2 text-sm text-slate-500 dark:text-slate-400">Log in to manage your tasks and notes.</p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
            <div>
                <label htmlFor="email" className="block text-sm font-medium leading-6 text-slate-900 dark:text-slate-200">Email address</label>
                <div className="mt-2 relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                        <span className="material-icons text-xl">mail_outline</span>
                    </div>
                    <input id="email" className="block w-full rounded-md border-0 py-2.5 pl-10 ring-1 ring-inset ring-slate-300 dark:ring-slate-700 placeholder:text-slate-400 focus:ring-2 focus:ring-inset focus:ring-primary dark:bg-slate-800 dark:text-white sm:text-sm sm:leading-6" placeholder="you@example.com" type="email" value={email} onChange={e => setEmail(e.target.value)} />
                </div>
            </div>
            
             <div>
                <label htmlFor="password" className="block text-sm font-medium leading-6 text-slate-900 dark:text-slate-200">Password</label>
                <div className="mt-2 relative rounded-md shadow-sm">
                    <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-slate-400">
                        <span className="material-icons text-xl">lock_outline</span>
                    </div>
                    <input id="password" className="block w-full rounded-md border-0 py-2.5 pl-10 ring-1 ring-inset ring-slate-300 dark:ring-slate-700 placeholder:text-slate-400 focus:ring-2 focus:ring-inset focus:ring-primary dark:bg-slate-800 dark:text-white sm:text-sm sm:leading-6" placeholder="Password" type="password" value={password} onChange={e => setPassword(e.target.value)} />
                </div>
            </div>

            {error && <p className="text-red-500 text-sm">{error}</p>}

            <button type="submit" disabled={loading} className="flex w-full justify-center rounded-md bg-primary px-3 py-2.5 text-sm font-semibold leading-6 text-white shadow-sm hover:bg-blue-600 transition-all disabled:opacity-50">
                {loading ? 'Logging in...' : 'Log in'}
            </button>
        </form>
        
<div className="mt-6 text-center">
    <p className="text-sm text-slate-500 dark:text-slate-400">
        Don't have an account?{' '}
        <button onClick={onSwitchToSignUp} className="font-semibold text-primary hover:text-primary-hover transition-colors">
            Sign up
        </button>
    </p>
</div>
      </div>
    </div>
  );
};

export default Login;
