// frontend/tests/pages/Login.test.tsx
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import Login from '../../pages/Login';
import { api } from '../../services/api';

vi.mock('../../services/api');

describe('Login Page', () => {
  const onLogin = vi.fn();
  const onSwitchToSignUp = vi.fn();

  beforeEach(() => {
    vi.resetAllMocks();
  });

  it('renders login form correctly', () => {
    render(<Login onLogin={onLogin} onSwitchToSignUp={onSwitchToSignUp} />);

    expect(screen.getByLabelText('Email address')).toBeInTheDocument();
    expect(screen.getByLabelText('Password')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Log in' })).toBeInTheDocument();
  });

  it('updates input fields on user input', async () => {
    render(<Login onLogin={onLogin} onSwitchToSignUp={onSwitchToSignUp} />);
    const emailInput = screen.getByLabelText('Email address');
    const passwordInput = screen.getByLabelText('Password');

    await userEvent.type(emailInput, 'test@example.com');
    await userEvent.type(passwordInput, 'password123');

    expect(emailInput).toHaveValue('test@example.com');
    expect(passwordInput).toHaveValue('password123');
  });

  it('calls onLogin on successful form submission', async () => {
    (api.login as vi.Mock).mockResolvedValueOnce({ access_token: 'test-token' });
    
    render(<Login onLogin={onLogin} onSwitchToSignUp={onSwitchToSignUp} />);
    const emailInput = screen.getByLabelText('Email address');
    const passwordInput = screen.getByLabelText('Password');
    const submitButton = screen.getByRole('button', { name: 'Log in' });

    await userEvent.type(emailInput, 'test@example.com');
    await userEvent.type(passwordInput, 'password123');
    await userEvent.click(submitButton);

    expect(api.login).toHaveBeenCalled();
    expect(onLogin).toHaveBeenCalled();
    expect(screen.queryByText(/error/i)).not.toBeInTheDocument();
  });

  it('displays error message on failed login', async () => {
    (api.login as vi.Mock).mockRejectedValueOnce(new Error('Invalid credentials'));

    render(<Login onLogin={onLogin} onSwitchToSignUp={onSwitchToSignUp} />);
    const emailInput = screen.getByLabelText('Email address');
    const passwordInput = screen.getByLabelText('Password');
    const submitButton = screen.getByRole('button', { name: 'Log in' });

    await userEvent.type(emailInput, 'test@example.com');
    await userEvent.type(passwordInput, 'wrongpassword');
    await userEvent.click(submitButton);

    expect(await screen.findByText('Invalid credentials')).toBeInTheDocument();
    expect(onLogin).not.toHaveBeenCalled();
  });

  it('calls onSwitchToSignUp when sign up link is clicked', async () => {
    render(<Login onLogin={onLogin} onSwitchToSignUp={onSwitchToSignUp} />);
    const signUpLink = screen.getByText('Sign up');
    
    await userEvent.click(signUpLink);

    expect(onSwitchToSignUp).toHaveBeenCalled();
  });
});
